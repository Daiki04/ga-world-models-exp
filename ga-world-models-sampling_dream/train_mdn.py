import math 
import torch
from torchvision import transforms
from multiprocessing import Lock
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import copy
import random
import os

# A: Action space, L: Latent space, R: Recurrent space, RED: Reduced size
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64

# 画像の前処理：1. PILイメージに変換、2. 64x64にリサイズ、3. テンソルに変換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


class RolloutGenerator(object):
    """ ロールアウトジェネレーター
    :attr device: VAE、MDRNN、コントローラーの実行に使用されるデバイス
    :attr time_limit: ロールアウトは最大time_limitタイムステップを持つ
    """

    def __init__(self, device, time_limit, discrete_VAE):
        """ vae、rnn、コントローラ、環境を構築 """
        ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64
        self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2
        
        self.device = device # デバイス

        self.time_limit = time_limit # タイムリミット

        self.discrete_VAE = discrete_VAE # Discrete VAEのオン／オフ

        #表現が離散的であるため、潜在ベクトルのサイズを大きく
        if (self.discrete_VAE):
            LSIZE = 128

        self.vae = VAE(3, LSIZE, 1024).to(self.device) # VAE：3チャンネル、潜在ベクトルサイズ、中間層サイズ
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device) # MDRNN：潜在ベクトルサイズ、アクションサイズ、中間層サイズ、混合ガウス分布の分布数
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(self.device) # コントローラー：潜在ベクトルサイズ、中間層サイズ、アクションサイズ

        file_path = self.get_file_path() # ファイルパスの取得
        mdn_file_path = self.mdn_path()
        self.load_solution(file_path, mdn_file_path)

    def get_file_path(self):
        dir_path = './results/'
        files = os.listdir(dir_path)
        files = [f for f in files if 'best_1_1_G' in f]

        num = -1
        idx = 0
        file = ''
        for idx, fi in enumerate(files):
            Gnum = fi.split('G')[1].split('.')[0]
            Gnum = int(Gnum)
            if Gnum > num:
                file = fi
                num = Gnum
                
        file_path = dir_path + file

        return file_path
    
    def mdn_path(self):
        dir_path = './results/'
        files = os.listdir(dir_path)
        files = [f for f in files if 'mdn_' in f]

        num = -1
        idx = 0
        checkpoint_file = ''
        for idx, fi in enumerate(files):
            Gnum = fi.split('_')[1].split('.')[0]
            Gnum = int(Gnum)
            if Gnum > num:
                checkpoint_file = fi
                num = Gnum
                
        checkpoint_path = dir_path + checkpoint_file
        return checkpoint_path

    def load_solution(self, filename, mdn_filename):
        """
        個体をロードする

        :args filename: ファイル名
        """

        s = torch.load(filename) # ファイルから個体をロード
        m = torch.load(mdn_filename) # ファイルから個体をロード

        self.vae.load_state_dict( s['vae']) # vaeに個体のvaeのパラメータをロード
        # self.controller.load_state_dict( s['controller']) # controllerに個体のcontrollerのパラメータをロード
        self.mdrnn.load_state_dict( s['mdrnn']) # mdrnnに個体のmdrnnのパラメータをロード
        self.mdrnn.gmm_linear.load_state_dict( m['mdn']) # mdrnnに個体のmdrnnのパラメータをロード

    def get_latent_mdn(self, mus, sigmas, logpi):
        """
        潜在ベクトルをサンプリングする

        :args mus: 潜在ベクトルの期待値 (bs, N_GAUSS, LSIZE)
        :args sigmas: 潜在ベクトルの標準偏差 (bs, N_GAUSS, LSIZE)
        :args logpi: 潜在ベクトルの混合ガウス分布の混合係数の対数 (bs, N_GAUSS)
        """
        # ① logpiを確率piに変換
        pi = torch.exp(logpi)
        pi /= torch.sum(pi, dim=1, keepdim=True)  # 各バッチごとに確率を正規化

        # サンプルバッチの初期化
        sampled_vectors = []

        # ② ③ 各バッチで分布を選択し、サンプリング
        for i in range(mus.size(0)):  # バッチサイズに対するループ
            chosen_component = torch.multinomial(pi[i], 1)  # ガウス分布の選択（確率に基づくサンプリング）
            chosen_component = chosen_component.squeeze()  # 1次元テンソルに変換
            sample = torch.normal(mus[i, chosen_component], torch.abs(sigmas[i, chosen_component]))
            sampled_vectors.append(sample)

        return torch.stack(sampled_vectors)

    def get_action_and_transition(self, latent_z, hidden):
        """ 行動を起こし、遷移

        VAEを用いて観測値を潜在状態に変換し、MDRNNを用いて次の潜在状態と次の隠れ状態の推定を行い、コントローラに対応するアクションを計算する。

        :args obs: current observation (1 x 32) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        # Discrete VAEの場合、離散化する

        action = self.controller(latent_z, hidden[0] ) # コントローラーによるアクションの計算

        mus, sigmas, logpi, rs, d, next_hidden = self.mdrnn(action, latent_z, hidden) # MDRNNによる次の潜在状態と次の隠れ状態の推定

        # 次の潜在状態のサンプリング
        next_latent_z = self.get_latent_mdn(mus, sigmas, logpi)
        reward = torch.sigmoid(rs) # 報酬のシグモイド関数による変換
        reward = torch.where(reward > 0.9, torch.tensor(1.0), torch.tensor(0.0))
        reward = reward.cpu().numpy()

        return action.squeeze().cpu().numpy(), next_hidden, next_latent_z, reward


    def do_rollout(self, render=False,  early_termination=True):
        """
        ロールアウトを実行する

        :args render: ロールアウトの描画を行うかどうか
        :args early_termination: ロールアウトを早期終了するかどうか
        """

        self.rollout_hidden = []
        self.rollout_latents = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_dones = []

        with torch.no_grad():
            
            self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2

            obs, _ = self.env.reset() # 環境のリセット
            done = False

            hidden = [
                torch.zeros(1, RSIZE).to(self.device) # 隠れ状態の初期化
                for _ in range(2)]
            
            self.rollout_hidden.append(hidden[0].cpu().numpy())

            neg_count = 0 # 負の報酬を受け取った回数

            cumulative = 0 # 累積報酬
            i = 0

            obs = transform(obs).unsqueeze(0).to(self.device) # 観測（画像）の前処理：obs(1, 3, 64, 64)
            _, latent_z, _, _ = self.vae(obs) 
            self.rollout_latents.append(latent_z.squeeze().cpu().numpy())

            while True:
                
                action, hidden,  next_latent_z, reward = self.get_action_and_transition(latent_z, hidden) # 行動を起こし、遷移
                self.rollout_actions.append(action)
                self.rollout_hidden.append(hidden[0].cpu().numpy())
                self.rollout_latents.append(next_latent_z.squeeze().cpu().numpy())
                self.rollout_rewards.append(reward)

                #報酬を得られなかった（コース外に出たなど）連続回数をカウント
                neg_count = neg_count+1 if reward <= 0.0 else 0 

                if render:
                    o = self.env.render("human") # 環境の描画
                
                #トレーニングのスピードアップのために、コース外の評価を行い，20time step以上コース外に出た場合はロールアウトを終了する
                if (neg_count>20 and early_termination):  
                    done = True
                
                cumulative += reward # 累積報酬の更新
                
                # ロールアウトの終了：タイムリミットに達した場合、早期終了した場合, 完了した場合
                if done or (early_termination and i > self.time_limit):
                    self.env.close()
                    self.rollout_latents.append(next_latent_z.squeeze().cpu().numpy())
                    
                    self.rollout_hidden = np.array(self.rollout_hidden)
                    self.rollout_latents = np.array(self.rollout_latents)
                    self.rollout_actions = np.array(self.rollout_actions)
                    self.rollout_rewards = np.array(self.rollout_rewards)
                    self.rollout_dones = np.array(self.rollout_dones)

                    rollout_current_hidden = self.rollout_hidden[:-1]
                    rollout_next_hidden = self.rollout_hidden[1:]
            
                    return cumulative, rollout_current_hidden, rollout_next_hidden, self.rollout_latents, self.rollout_actions, self.rollout_rewards, self.rollout_dones
                
                latent_z = next_latent_z
                i += 1

