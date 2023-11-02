from torchvision import transforms
import neat
import numpy as np
import gymnasium as gym
from neat.config import Config
from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
from neat.stagnation import DefaultStagnation
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation
from neat.population import PopulationVMC
import warnings
import multiprocessing
import os
import time

import vizualize

warnings.simplefilter('ignore')

time_limit = 1000 # タイムリミット
env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2

# A: Action space, L: Latent space, R: Recurrent space, RED: Reduced size
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64

# 画像の前処理：1. PILイメージに変換、2. 64x64にリサイズ、3. テンソルに変換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

def eval_genomes(genomes, configs):
    genomeV, genomeM, genomeC = genomes
    configV, configM, configC = configs

    for _, gV in genomeV:
        gV.fitness = 0.0
    for _, gM in genomeM:
        gM.fitness = 0.0
    for _, gC in genomeC:
        gC.fitness = 0.0

    for idV, gV in genomeV:
        for idM, gM in genomeM:
            for idC, gC in genomeC:
                score = 0
                for _ in range(3):
                    time_count = 0
                    netV = neat.nn.FeedForwardNetwork.create(gV, configV) # VAE
                    netM = neat.nn.RecurrentNetwork.create(gM, configM) # MDRNN
                    netC = neat.nn.FeedForwardNetwork.create(gC, configC)

                    cumulative = 0

                    hidden = np.zeros(RSIZE) # 隠れ状態

                    env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=False) # 環境：CarRacing-v2
                    obs, _ = env.reset() # 環境をリセット
                    neg_reward = 0

                    while True:
                        obs = transform(obs).unsqueeze(0) # 画像を前処理
                        obs = obs.flatten().detach().numpy() # 画像をベクトルに変換

                        latent_mu_sigma= netV.activate(obs)
                        latent_mu_sigma = np.array(latent_mu_sigma)
                        latent_mu = latent_mu_sigma[:LSIZE]
                        latent_sigma = latent_mu_sigma[LSIZE:]
                        latent = latent_mu + latent_sigma * np.random.normal(size=latent_mu.shape)
                        C_input = np.concatenate([latent, hidden])
                        action = netC.activate(C_input)
                        # action: [steering, gas, brake], steering: [-1, 1], gas: [0, 1], brake: [0, 1]
                        action[0] = action[0] * 2 - 1 # steering
                        M_input = np.concatenate([action, latent])
                        next_hidden = netM.activate(M_input)

                        obs, reward, done, _, _ = env.step(action) # 行動を起こす
                        cumulative += reward

                        if reward < 0:
                            neg_reward += 1
                        else:
                            neg_reward = 0

                        if time_count > time_limit or neg_reward > 15:
                            done = True
                        
                        if done:
                            env.close()
                            break

                        hidden = next_hidden
                        time_count += 1
                    score += cumulative
                score /= 3

                if gM.fitness < score:
                    gM.fitness = score
                if gC.fitness < score:
                    gC.fitness = score
                if gV.fitness < score:
                    gV.fitness = score
                print(f"id:[{idV}, {idM}, {idC}], fitness: {score}")


if __name__ == '__main__':
    # Load configuration.
    confV = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, './config/view.cfg')
    confM = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, './config/memory.cfg')
    confC = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, './config/controller.cfg')

    # Create population
    p = PopulationVMC([confV, confM, confC])

    # run
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)
    bV, bM, bC, histV, histM, histC = p.run(eval_genomes, 50)

    # make result directory
    result_dir = './result' + str(int(time.time()))
    os.mkdir(result_dir)

    # save best genome
    np.save(result_dir + '/bestV.npy', bV)
    np.save(result_dir + '/bestM.npy', bM)
    np.save(result_dir + '/bestC.npy', bC)

    # save history
    np.save(result_dir + '/histV.npy', histV)
    np.save(result_dir + '/histM.npy', histM)
    np.save(result_dir + '/histC.npy', histC)

    # draw and save best genome
    # vizualize.draw_net(confV, bV, filename=result_dir + '/bestV.gv')
    # vizualize.draw_net(confM, bM, filename=result_dir + '/bestM.gv')
    # vizualize.draw_net(confC, bC, filename=result_dir + '/bestC.gv')

    # print best genome
    print(bV)
    print(bM)
    print(bC)