from torchvision import transforms
import neat
import numpy as np
import gym
from neat.config import Config
from neat.genome import DefaultGenome
from neat.reproduction import DefaultReproduction
from neat.stagnation import DefaultStagnation
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation
from neat.population import PopulationVMC
import warnings
import os
import time

import vizualize

warnings.simplefilter('ignore')

time_limit = 2000 # タイムリミット
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
                    M_input = np.concatenate([action, latent])
                    hidden = netM.activate(M_input)

                    obs, reward, done, _, _ = env.step(action) # 行動を起こす
                    cumulative += reward

                    if reward < 0:
                        neg_reward += 1
                    else:
                        neg_reward = 0

                    if time_count > time_limit or neg_reward > 20:
                        done = True
                    
                    if done:
                        env.close()
                        if gM.fitness < cumulative:
                            gM.fitness = cumulative
                        if gC.fitness < cumulative:
                            gC.fitness = cumulative
                        if gV.fitness < cumulative:
                            gV.fitness = cumulative
                        break

                    time_count += 1
                print(f"id:[{idV}, {idM}, {idC}], fitness: {cumulative}")


if __name__ == '__main__':
    # Load configuration.
    confV = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, './config/view.cfg')
    confM = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, './config/memory.cfg')
    confC = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, './config/controller.cfg')

    # Create population
    p = PopulationVMC([confV, confM, confC])

    # run
    bV, bM, bC = p.run(eval_genomes, 1000)

    # make result directory
    result_dir = './result' + str(int(time.time()))
    os.mkdir(result_dir)

    # save best genome
    np.save(result_dir + '/bestV.npy', bV)
    np.save(result_dir + '/bestM.npy', bM)
    np.save(result_dir + '/bestC.npy', bC)

    # draw and save best genome
    # vizualize.draw_net(confV, bV, filename=result_dir + '/bestV.gv')
    # vizualize.draw_net(confM, bM, filename=result_dir + '/bestM.gv')
    # vizualize.draw_net(confC, bC, filename=result_dir + '/bestC.gv')

    # print best genome
    print(bV)
    print(bM)
    print(bC)