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
        self.load_solution(file_path) # 個体のロード

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

    def load_solution(self, filename):
        """
        個体をロードする

        :args filename: ファイル名
        """

        s = torch.load(filename) # ファイルから個体をロード

        self.vae.load_state_dict( s['vae']) # vaeに個体のvaeのパラメータをロード
        self.controller.load_state_dict( s['controller']) # controllerに個体のcontrollerのパラメータをロード
        self.mdrnn.load_state_dict( s['mdrnn']) # mdrnnに個体のmdrnnのパラメータをロード

    def get_action_and_transition(self, obs, hidden):
        """ 行動を起こし、遷移

        VAEを用いて観測値を潜在状態に変換し、MDRNNを用いて次の潜在状態と次の隠れ状態の推定を行い、コントローラに対応するアクションを計算する。

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_z, _, _ = self.vae(obs) # latent_z: 潜在ベクトルの期待値

        # Discrete VAEの場合、離散化する
        if (self.discrete_VAE):  

            bins=np.array([-1.0,0.0,1.0]) # 

            latent_z = torch.tanh(latent_z) # 潜在ベクトルの期待値を[-1,1]に変換
            newdata=bins[np.digitize(latent_z,bins[1:])]+1 # latent_zをdigitizeし、[0,1]に変換（0以下：0, 0より大きい：1), 1を足すことで[0, 1]に変換

            latent_z = torch.from_numpy(newdata).float() # latent_zをtorch tensorに変換
        
        self.rollout_latents.append(latent_z.squeeze().cpu().numpy()) # ロールアウトの潜在状態の追加

        action = self.controller(latent_z, hidden[0] ) # コントローラーによるアクションの計算

        mus, sigmas, logpi, rs, d, next_hidden = self.mdrnn(action, latent_z, hidden) # MDRNNによる次の潜在状態と次の隠れ状態の推定

        return action.squeeze().cpu().numpy(), next_hidden

    def get_random_action_and_transition(self, obs, hidden):
        """ 行動を起こし、遷移

        VAEを用いて観測値を潜在状態に変換し、MDRNNを用いて次の潜在状態と次の隠れ状態の推定を行い、コントローラに対応するアクションを計算する。

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_z, _, _ = self.vae(obs) # latent_z: 潜在ベクトルの期待値
        
        self.rollout_latents.append(latent_z.squeeze().cpu().numpy()) # ロールアウトの潜在状態の追加

        steering = np.random.uniform(-1, 1)
        acceleration = np.random.uniform(0, 1)
        brake = np.random.uniform(0, 1)
        action = np.array([steering, acceleration, brake])
        action = torch.from_numpy(action).float().unsqueeze(0).to(self.device)

        mus, sigmas, logpi, rs, d, next_hidden = self.mdrnn(action, latent_z, hidden) # MDRNNによる次の潜在状態と次の隠れ状態の推定

        return action.squeeze().cpu().numpy(), next_hidden


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

        if np.random.rand() < 0.8:
            with torch.no_grad():
                
                self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2
                done = False
    
                obs, _ = self.env.reset() # 環境のリセット
    
                hidden = [
                    torch.zeros(1, RSIZE).to(self.device) # 隠れ状態の初期化
                    for _ in range(2)]
                
                self.rollout_hidden.append(hidden[0].cpu().numpy())
    
                neg_count = 0 # 負の報酬を受け取った回数
    
                cumulative = 0 # 累積報酬
                i = 0
                while True:
                    obs = transform(obs).unsqueeze(0).to(self.device) # 観測（画像）の前処理：obs(1, 3, 64, 64)
                    
                    action, hidden = self.get_action_and_transition(obs, hidden) # 行動を起こし、遷移：action(1, ASIZE), hidden(1, RSIZE)
                    self.rollout_actions.append(action)
                    self.rollout_hidden.append(hidden[0].cpu().numpy())
                    #Steering: Real valued in [-1, 1] 
                    #Gas: Real valued in [0, 1]
                    #Break: Real valued in [0, 1]
    
                    obs, reward, done, _, _ = self.env.step(action) # 行動を実行し、報酬を受け取る：obs(3, 64, 64), reward, done, info
                    self.rollout_rewards.append(reward)
                    self.rollout_dones.append(done)
                    #報酬を得られなかった（コース外に出たなど）連続回数をカウント
                    neg_count = neg_count+1 if reward < 0.0 else 0   
    
                    if render:
                        o = self.env.render("human") # 環境の描画
                    
                    #トレーニングのスピードアップのために、コース外の評価を行い，20time step以上コース外に出た場合はロールアウトを終了する
                    
                    if (neg_count>20 and early_termination):  
                        done = True
                    
                    cumulative += reward # 累積報酬の更新
                    
                    # ロールアウトの終了：タイムリミットに達した場合、早期終了した場合, 完了した場合
                    if done or (early_termination and i > self.time_limit):
                        self.env.close()
                        obs = transform(obs).unsqueeze(0).to(self.device)
                        _, latent_z, _, _ = self.vae(obs)
                        self.rollout_latents.append(latent_z.squeeze().cpu().numpy())
                        
                        self.rollout_hidden = np.array(self.rollout_hidden)
                        self.rollout_latents = np.array(self.rollout_latents)
                        self.rollout_actions = np.array(self.rollout_actions)
                        self.rollout_rewards = np.array(self.rollout_rewards)
                        self.rollout_dones = np.array(self.rollout_dones)
    
                        rollout_current_hidden = self.rollout_hidden[:-1]
                        rollout_next_hidden = self.rollout_hidden[1:]
                
                        return cumulative, rollout_current_hidden, rollout_next_hidden, self.rollout_latents, self.rollout_actions, self.rollout_rewards, self.rollout_dones
    
                    i += 1
        else:
            with torch.no_grad():
                
                self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2

                obs, _ = self.env.reset() # 環境のリセット

                hidden = [
                    torch.zeros(1, RSIZE).to(self.device) # 隠れ状態の初期化
                    for _ in range(2)]
                
                self.rollout_hidden.append(hidden[0].cpu().numpy())

                neg_count = 0 # 負の報酬を受け取った回数
                done = False

                cumulative = 0 # 累積報酬
                i = 0
                while True:
                    obs = transform(obs).unsqueeze(0).to(self.device) # 観測（画像）の前処理：obs(1, 3, 64, 64)
                    
                    action, hidden = self.get_random_action_and_transition(obs, hidden) # 行動を起こし、遷移：action(1, ASIZE), hidden(1, RSIZE)
                    self.rollout_actions.append(action)
                    self.rollout_hidden.append(hidden[0].cpu().numpy())
                    #Steering: Real valued in [-1, 1] 
                    #Gas: Real valued in [0, 1]
                    #Break: Real valued in [0, 1]

                    obs, reward, done, _, _ = self.env.step(action) # 行動を実行し、報酬を受け取る：obs(3, 64, 64), reward, done, info
                    self.rollout_rewards.append(reward)
                    self.rollout_dones.append(done)
                    #報酬を得られなかった（コース外に出たなど）連続回数をカウント
                    neg_count = neg_count+1 if reward < 0.0 else 0   

                    if render:
                        o = self.env.render("human") # 環境の描画
                    
                    #トレーニングのスピードアップのために、コース外の評価を行い，20time step以上コース外に出た場合はロールアウトを終了する
                    # if (neg_count>20 and early_termination):  
                    #     done = True
                    
                    cumulative += reward # 累積報酬の更新
                    
                    # ロールアウトの終了：タイムリミットに達した場合、早期終了した場合, 完了した場合
                    if done or (early_termination and i > self.time_limit):
                        self.env.close()
                        obs = transform(obs).unsqueeze(0).to(self.device)
                        _, latent_z, _, _ = self.vae(obs)
                        self.rollout_latents.append(latent_z.squeeze().cpu().numpy())
                        
                        self.rollout_hidden = np.array(self.rollout_hidden)
                        self.rollout_latents = np.array(self.rollout_latents)
                        self.rollout_actions = np.array(self.rollout_actions)
                        self.rollout_rewards = np.array(self.rollout_rewards)
                        self.rollout_dones = np.array(self.rollout_dones)

                        rollout_current_hidden = self.rollout_hidden[:-1]
                        rollout_next_hidden = self.rollout_hidden[1:]
                
                        return cumulative, rollout_current_hidden, rollout_next_hidden, self.rollout_latents, self.rollout_actions, self.rollout_rewards, self.rollout_dones

                    i += 1

    def get_rollout(self, num_rollouts, dir='./rollouts/'):
        """
        ロールアウトを実行する

        :args num_rollouts: ロールアウトの回数
        """

        rollouts = []
        for i in range(num_rollouts):
            cumlatives, rollout_current_hidden, rollout_next_hidden, rollout_latents, rollout_actions, rollout_rewards, rollout_dones = self.do_rollout()
            cumlatives = np.array(cumlatives)
            rollout_current_hidden = np.array(rollout_current_hidden)
            rollout_next_hidden = np.array(rollout_next_hidden)
            rollout_latents = np.array(rollout_latents)
            rollout_actions = np.array(rollout_actions)
            rollout_rewards = np.array(rollout_rewards)
            rollout_dones = np.array(rollout_dones)
            np.savez_compressed(os.path.join(dir, 'rollout_{}.npz'.format(i)), cumlatives=cumlatives, rollout_current_hidden=rollout_current_hidden, rollout_next_hidden=rollout_next_hidden, rollout_latents=rollout_latents, rollout_actions=rollout_actions, rollout_rewards=rollout_rewards, rollout_dones=rollout_dones)

    def load_rollout(self, dir='./rollouts/'):
        """
        ロールアウトをロードする
        """
        files = os.listdir(dir)
        files = [f for f in files if 'rollout' in f]
        rollouts_cumlatives = []
        rollouts_current_hidden = []
        rollouts_next_hidden = []
        rollouts_latents = []
        rollouts_actions = []
        rollouts_rewards = []
        rollouts_dones = []

        for f in files:
            data = np.load(os.path.join(dir, f))
            rollouts_cumlatives.append(torch.from_numpy(data['cumlatives']).float())
            rollouts_current_hidden.append(torch.from_numpy(data['rollout_current_hidden']).float())
            rollouts_next_hidden.append(torch.from_numpy(data['rollout_next_hidden']).float())
            rollouts_latents.append(torch.from_numpy(data['rollout_latents']).float())
            rollouts_actions.append(torch.from_numpy(data['rollout_actions']).float())
            rollouts_rewards.append(torch.from_numpy(data['rollout_rewards']).float())
            rollouts_dones.append(torch.from_numpy(data['rollout_dones']).float())
        
        rollouts_cumlatives = torch.Tensor(rollouts_cumlatives)
        rollouts_current_hidden = torch.Tensor(rollouts_current_hidden)
        

        return rollouts_cumlatives, rollouts_current_hidden, rollouts_next_hidden, rollouts_latents, rollouts_actions, rollouts_rewards, rollouts_dones
