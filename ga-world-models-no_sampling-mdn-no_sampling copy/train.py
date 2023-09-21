import math 
import torch
from torchvision import transforms
from multiprocessing import Lock
import numpy as np
from models import MDRNNCell, VAE, Controller
import gym
import copy
import random

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

        self.vae = VAE(3, LSIZE, 1024) # VAE：3チャンネル、潜在ベクトルサイズ、中間層サイズ
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5) # MDRNN：潜在ベクトルサイズ、アクションサイズ、中間層サイズ、混合ガウス分布の分布数
        self.controller = Controller(LSIZE, RSIZE, ASIZE) # コントローラー：潜在ベクトルサイズ、中間層サイズ、アクションサイズ


    def get_action_and_transition(self, obs, hidden, latent_mu_prediction):
        """ 行動を起こし、遷移

        VAEを用いて観測値を潜在状態に変換し、MDRNNを用いて次の潜在状態と次の隠れ状態の推定を行い、コントローラに対応するアクションを計算する。

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs) # latent_mu: 潜在ベクトルの期待値

        # Discrete VAEの場合、離散化する
        if (self.discrete_VAE):  

            bins=np.array([-1.0,0.0,1.0]) # 

            latent_mu = torch.tanh(latent_mu) # 潜在ベクトルの期待値を[-1,1]に変換
            newdata=bins[np.digitize(latent_mu,bins[1:])]+1 # latent_muをdigitizeし、[0,1]に変換（0以下：0, 0より大きい：1), 1を足すことで[0, 1]に変換

            latent_mu = torch.from_numpy(newdata).float() # latent_muをtorch tensorに変換

        if latent_mu_prediction is None: # 最初のステップの場合
            action = self.controller(latent_mu, hidden[0] ) # コントローラーによるアクションの計算
        else:
            action = self.controller(latent_mu_prediction[0], hidden[0] ) # コントローラーによるアクションの計算

        mus, sigmas, logpi, rs, d, next_hidden = self.mdrnn(action, latent_mu, hidden) # MDRNNによる次の潜在状態と次の隠れ状態の推定

        next_latent_mu_prediction = self.select_gaussian(logpi, mus) # 次の潜在状態の推定値をガウス分布からサンプリング

        return action.squeeze().cpu().numpy(), next_hidden, next_latent_mu_prediction # アクション、次の隠れ状態、次の潜在状態の推定値を返す
    
    def select_gaussian(self, logpi_nlat, mu_nlat):
        # logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) tensor
        # mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) tensor

        # N_GAUSSから確率分布をサンプリングする
        # まず、確率分布からサンプリングするための確率を計算します
        pi_probs = torch.softmax(logpi_nlat, dim=2)  # (SEQ_LEN, BSIZE, N_GAUSS)

        # 各要素ごとにサンプリングを実行
        sampled_gaussians = torch.randint(5, size=(logpi_nlat.size(0), logpi_nlat.size(1)))  # (SEQ_LEN, BSIZE)

        # サンプリングしたガウス分布から対応するmuを取得
        selected_mu = torch.gather(mu_nlat, dim=2, index=sampled_gaussians.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, mu_nlat.size(3)))  # (SEQ_LEN, BSIZE, 1, LSIZE)

        # 不要な次元を削除
        selected_mu = selected_mu.squeeze(2)  # (SEQ_LEN, BSIZE, LSIZE)

        return selected_mu


    def do_rollout(self, render=False,  early_termination=True):
        """
        ロールアウトを実行する

        :args render: ロールアウトの描画を行うかどうか
        :args early_termination: ロールアウトを早期終了するかどうか
        """


        with torch.no_grad():
            
            self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2

            obs, _ = self.env.reset() # 環境のリセット

            hidden = [
                torch.zeros(1, RSIZE)#.to(self.device) # 隠れ状態の初期化
                for _ in range(2)]

            neg_count = 0 # 負の報酬を受け取った回数

            cumulative = 0 # 累積報酬
            i = 0
            while True:
                obs = transform(obs).unsqueeze(0)#.to(self.device) # 観測（画像）の前処理：obs(1, 3, 64, 64)
                
                if i == 0:
                    action, hidden, latent_mu_prediction = self.get_action_and_transition(obs, hidden, None)
                else:
                    action, hidden, latent_mu_prediction = self.get_action_and_transition(obs, hidden, latent_mu_prediction) # 行動を起こし、遷移：action(1, ASIZE), hidden(1, RSIZE)
                #Steering: Real valued in [-1, 1] 
                #Gas: Real valued in [0, 1]
                #Break: Real valued in [0, 1]

                obs, reward, done, _, _ = self.env.step(action) # 行動を実行し、報酬を受け取る：obs(3, 64, 64), reward, done, info
                
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
                    return cumulative, None

                i += 1


# 並列処理における適応度評価
def fitness_eval_parallel(pool, r_gen, early_termination=True):#, controller_parameters):
    return pool.apply_async(r_gen.do_rollout, args=(False, early_termination) )


class GAIndividual():
    '''
    GAの個体クラス

    multi = マルチプロセッシングのオン／オフを切り替えるフラグ 
    '''
    def __init__(self, device, time_limit, setting, multi=True, discrete_VAE = False):
        '''
        コンストラクタ―
        '''

        self.device = device # デバイス
        self.time_limit = time_limit # タイムリミット
        self.multi = multi # マルチプロセスのオン／オフ
        self.discrete_VAE = discrete_VAE # Discrete VAEのオン／オフ

        self.mutation_power = 0.01 # 突然変異時の変化量
            
        self.setting = setting # 0: MUT-ALL, 1: MUT-MOD

        self.r_gen = RolloutGenerator(device, time_limit, discrete_VAE) # ロールアウトジェネレーター
        #self.r_gen.discrete_VAE = self.discrete_VAE

        self.async_results = [] # 非同期結果
        self.calculated_results = {} # 計算結果

    def run_solution(self, pool, evals=5, early_termination=True, force_eval=False):
        """
        問題を解き、適応度を計算する

        :args pool: マルチプロセスのプール
        :args evals: 評価回数
        :args early_termination: ロールアウトを早期終了するかどうか
        :args force_eval: 計算結果を削除するかどうか
        """

        # force_eval = Trueの場合、最後尾の計算結果を削除する
        if force_eval:
            self.calculated_results.pop(evals, None)

        # 最後尾の計算結果が存在する場合，計算を行わない
        if (evals in self.calculated_results.keys()): #Already caculated results
            return

        # 非同期でロールアウトを実行する
        self.async_results = []

        for i in range(evals):

            if self.multi:
                # 並列処理でevals回問題を実行し、結果をasync_resultsに格納する
                self.async_results.append(fitness_eval_parallel(pool, self.r_gen, early_termination))#, self.controller_parameters) )
            else:
                # 並列処理を行わないでevals回問題を実行し、結果をasync_resultsに格納する
                self.async_results.append(self.r_gen.do_rollout(False, early_termination)) 


    def evaluate_solution(self, evals):
        """
        個体を評価する

        :args evals: 評価回数
        """

        # 評価回数分の計算結果が存在する場合、計算結果を返す
        if (evals in self.calculated_results.keys()): #Already calculated?
            mean_fitness, std_fitness = self.calculated_results[evals] # 計算結果を取得

        else:
            # 結果を取得
            if self.multi:
                # 並列処理の場合
                results = [t.get()[0] for t in self.async_results]
            else:
                # 並列処理を行わない場合
                results = [t[0] for t in self.async_results]

            # 結果の平均と標準偏差を計算
            mean_fitness = np.mean ( results )
            std_fitness = np.std( results )

            # 計算結果を格納
            self.calculated_results[evals] = (mean_fitness, std_fitness)

        # 適応度：平均値のマイナス，小さいほど良い
        self.fitness = -mean_fitness

        return mean_fitness, std_fitness


    def load_solution(self, filename):
        """
        個体をロードする

        :args filename: ファイル名
        """

        s = torch.load(filename) # ファイルから個体をロード

        self.r_gen.vae.load_state_dict( s['vae']) # vaeに個体のvaeのパラメータをロード
        self.r_gen.controller.load_state_dict( s['controller']) # controllerに個体のcontrollerのパラメータをロード
        self.r_gen.mdrnn.load_state_dict( s['mdrnn']) # mdrnnに個体のmdrnnのパラメータをロード

    
    def clone_individual(self):
        """
        個体を複製する
        """
        child_solution = GAIndividual(self.device, self.time_limit, self.setting, multi=True, discrete_VAE = self.discrete_VAE) # 個体クラスを生成
        child_solution.multi = self.multi # マルチプロセスのオン／オフをコピー

        child_solution.fitness = self.fitness # 適応度をコピー

        child_solution.r_gen.controller = copy.deepcopy (self.r_gen.controller) # controllerをコピー
        child_solution.r_gen.vae = copy.deepcopy (self.r_gen.vae) # vaeをコピー
        child_solution.r_gen.mdrnn = copy.deepcopy (self.r_gen.mdrnn) # mdrnnをコピー
        
        return child_solution # 複製した個体を返す
    
    def mutate_params(self, params):
        """
        パラメータを突然変異させる

        :args params: 突然変異対称のパラメータ
        """
        for key in params: 
               # パラメーターに対して，（正規分布に従う乱数）*（突然変異の変異量）を加える
               params[key] += torch.from_numpy( np.random.normal(0, 1, params[key].size()) * self.mutation_power).float()

    def mutate(self):
        """
        突然変異を行う
        """
        if self.setting == 0: #Mutate controller, VAE and MDRNN. Normal deep neuroevolution

            # すべてのモジュールのパラメータを突然変異させる
            self.mutate_params(self.r_gen.controller.state_dict())
            self.mutate_params(self.r_gen.vae.state_dict())
            self.mutate_params(self.r_gen.mdrnn.state_dict())

        if self.setting == 1: #Mutate controller, VAE or mdrnn
            c = np.random.randint(0,3)

            # ランダムに選択したモジュールのパラメータを突然変異させる
            if c==0:
                self.mutate_params(self.r_gen.vae.state_dict())
            elif c==1:
                self.mutate_params(self.r_gen.mdrnn.state_dict() )
            else:
                self.mutate_params(self.r_gen.controller.state_dict())
