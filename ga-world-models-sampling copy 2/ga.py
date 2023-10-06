import sys, random
import numpy as np
import pickle # オブジェクトのロードと保存を行う
import torch # PyTorch
import time # 時間計測
import math # 数学関数
from os.path import join, exists # ファイルのパスを結合, ファイルの存在確認
import multiprocessing # マルチプロセス
import gc # ガベージコレクション
import copy # オブジェクトのコピー

from multiprocessing import set_start_method # マルチプロセスの開始方法を設定
# set_start_method('forkserver', force=True) # マルチプロセスの開始方法を設定：親プロセスのインタプリタ, オブジェクト, モジュールなどを全てコピーする
set_start_method('spawn', force=True) # マルチプロセスの開始方法を設定：親プロセスのインタプリタ, オブジェクト, モジュールなどを全てコピーする


class GA:
    def __init__(self, elite_evals, top, threads, timelimit, pop_size, setting, discrete_VAE):
        '''
        コンストラクタ―

        :args elite_evals: 上位個体の評価回数
        :args top: 再評価を行う上位個体数
        :args threads: スレッド数
        :args timelimit: 時間ステップの終了条件
        :args pop_size: 個体数
        :args setting: 設定
        :args discrete_VAE: 離散VAEのオン／オフ
        '''
        self.top  = top  # 再評価を行う上位個体数
        self.elite_evals = elite_evals  #上位個体の評価回数

        self.pop_size = pop_size # 個体数

        self.threads = threads # スレッド数
        multi_process = threads>1 # マルチプロセスの有無

        self.truncation_threshold = int(pop_size/2)  # トーナメント選択の閾値

        from train import GAIndividual # GAの個体クラスをインポート

        self.P = [] # 個体リスト

        # pop_size個の個体を生成し，個体リストに格納
        for i in range(pop_size):
            self.P.append(GAIndividual('cpu', timelimit, setting, multi= multi_process, discrete_VAE=discrete_VAE ) )

        
    def run(self, max_generations, filename, folder):
        """
        最適化の実行

        :args max_generations: 最大世代数
        :args filename: ファイル名
        :args folder: フォルダ名
        """

        Q = []
        
        max_fitness = -sys.maxsize # 適応度の最良値，適応度の最小値で初期化

        fitness_file = open(folder+"/fitness_"+filename+".txt", 'a') # 適応度のファイルを追記モードで開く

        ind_fitness_file = open(folder+"/individual_fitness_"+filename+".txt", 'a') # 個体の適応度のファイルを追記モードで開く
        
        i = 0
        P = self.P

        pop_name = folder+"/pop_"+filename+".p" # 個体群を保存するファイル名

        # 個体群を保存するファイルが存在する場合，個体群を読み込む
        if exists( pop_name ):
            """
            途中から再開する場合
            """
            pop_tmp = torch.load(pop_name) # 個体群を読み込む

            print("Loading existing population ",pop_name, len(pop_tmp)) # 個体群の読み込み結果を表示

            idx = 0
            for s in pop_tmp:
                 P[idx].r_gen.vae.load_state_dict ( s['vae'].copy() ) # vaeのパラメータをロード
                 P[idx].r_gen.controller.load_state_dict ( s['controller'].copy() ) # controllerのパラメータをロード
                 P[idx].r_gen.mdrnn.load_state_dict ( s['mdrnn'].copy() ) # mdrnnのパラメータをロード

                 i = s['generation'] + 1 # 完了済みの世代数を取得
                 idx+=1
                 

        while (True): 
            pool = multiprocessing.Pool(self.threads) # マルチプロセスのプールを生成

            start_time = time.time() # 時間計測開始

            print("Generation ", i) # 世代数を表示
            sys.stdout.flush() # 標準出力をフラッシュ

            print("Evaluating individuals: ",len(P) ) # 評価個体数を表示
            for s in P:  
                s.run_solution(pool, 1, force_eval=True) # 問題を解き、適応度を計算（evals=1）

            fitness = [] # 適応度リスト

            for s in P:
                s.is_elite = False # エリートフラグをリセット
                f, _ = s.evaluate_solution(1) # 適応度を計算（evals=1）
                fitness += [f] # 適応度をリストに追加

            self.sort_objective(P)

            max_fitness_gen = -sys.maxsize #この世代最高のフィットネスを記録する，最小値で初期化

            print("Evaluating elites: ", self.top) # エリート個体数を表示

            for k in range(self.top):      
                P[k].run_solution(pool, self.elite_evals) # 問題を解き、適応度を計算（evals=elite_evals）
            
            for k in range(self.top):

                f, _ = P[k].evaluate_solution(self.elite_evals) # 適応度を計算（evals=elite_evals）

                # この世代で最も適応度が高い個体を保存
                if f>max_fitness_gen:
                    max_fitness_gen = f
                    elite = P[k]

                # 全世代で最も適応度が高い個体を保存
                if f > max_fitness: #best fitness ever found
                    max_fitness = f
                    print("\tFound new champion ", max_fitness )

                    best_ever = P[k]
                    sys.stdout.flush()
                    
                    torch.save({'vae': elite.r_gen.vae.state_dict(), 'controller': elite.r_gen.controller.state_dict(), 'mdrnn':elite.r_gen.mdrnn.state_dict(), 'fitness':f}, "{0}/best_{1}G{2}.p".format(folder, filename, i))

            elite.is_elite = True  # この世代の最良個体にエリートフラグを設定

            sys.stdout.flush()

            pool.close()

            Q = []

            # トーナメント選択に使用しない個体を削除
            if len(P) > self.truncation_threshold-1:
                del P[self.truncation_threshold-1:]

            P.append(elite) # 全世代での最良個体を個体リストに追加

            save_pop = []

            # 現世代の上位個体の適応度をファイルに保存
            for s in P:
                 ind_fitness_file.write( "Gen\t%d\tFitness\t%f\n" % (i, -s.fitness )  )  
                 ind_fitness_file.flush()

                 save_pop += [{'vae': s.r_gen.vae.state_dict(), 'controller': s.r_gen.controller.state_dict(), 'mdrnn':s.r_gen.mdrnn.state_dict(), 'fitness':fitness, 'generation':i}]
                 
            # 25世代ごとに個体群をファイルに保存
            if (i % 25 == 0):
                print("saving population")
                torch.save(save_pop, folder+"/pop_"+filename+".p")
                print("done")

            print("Creating new population ...", len(P))
            Q = self.make_new_pop(P)

            P.extend(Q) # 新しい個体を個体リストに追加

            elapsed_time = time.time() - start_time # 時間計測終了

            print( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" % (i, np.mean(fitness), max_fitness_gen, max_fitness, elapsed_time) )  # python will convert \n to os.linesep

            fitness_file.write( "%d\tAverage\t%f\tMax\t%f\tMax ever\t%f\tTime\t%f\n" % (i, np.mean(fitness), max_fitness_gen, max_fitness,  elapsed_time) )  # python will convert \n to os.linesep
            fitness_file.flush()

            # 最大世代数に達した場合，最適化を終了
            if (i > max_generations):
                break

            gc.collect() # ガベージコレクション

            i += 1 # 世代を更新

        print("Testing best ever: ")
        pool = multiprocessing.Pool(self.threads)


        best_ever.run_solution(pool, 100, early_termination=False, force_eval = True) # 全世代で最も適応度が高い個体を100回実行
        avg_f, sd = best_ever.evaluate_solution(100) # 全世代で最も適応度が高い個体の適応度を計算（evals=100）
        print(avg_f, sd)
        
        fitness_file.write( "Test\t%f\t%f\n" % (avg_f, sd) ) 

        fitness_file.close()

        ind_fitness_file.close()

                                
    def sort_objective(self, P):
        """
        個体を適応度でソートする
        indexが小さいほど適応度が高い

        :args P: 個体リスト
        """
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if s1.fitness > s2.fitness:
                    P[j - 1] = s2
                    P[j] = s1
                    

    def make_new_pop(self, P):
        '''
        Pの子孫である新しい集団Qを作る 

        :args P: 現世代の個体リスト
        '''
        Q = []
        
        while len(Q) < self.truncation_threshold:
            selected_solution = None
            
            s1 = random.choice(P)
            s2 = s1
            while s1 == s2:
                s2 = random.choice(P)

            if s1.fitness < s2.fitness: #低いほど良い
                selected_solution = s1
            else:
                selected_solution  = s2

            if s1.is_elite:  #エリート個体は確実に選択
                selected_solution = s1
            elif s2.is_elite:  
                selected_solution = s2
            
            child_solution = selected_solution.clone_individual() # 選択した個体を複製
            child_solution.mutate() # 個体を突然変異させる

            # 重複していない場合，子孫を個体リストに追加
            if (not child_solution in Q):    
                Q.append(child_solution)
        
        return Q
        
