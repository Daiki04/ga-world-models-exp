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
from neat.population import Population
import warnings
import multiprocessing
import os
import time

import vizualize

warnings.simplefilter('ignore')

time_limit = 1000  # タイムリミット
env = gym.make('CarRacing-v2', render_mode='rgb_array',
               domain_randomize=False)  # 環境：CarRacing-v2

# A: Action space, L: Latent space, R: Recurrent space, RED: Reduced size
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64

# 画像の前処理：1. PILイメージに変換、2. 64x64にリサイズ、3. テンソルに変換
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = 0
        total_reward = 0
        obs, _ = env.reset()  # 環境をリセット
        net = neat.nn.RecurrentNetwork.create(genome, config)

        for _ in range(5):
            time_count = 0
            cumulative = 0
            neg_reward = 0
            obs, _ = env.reset()

            while True:
                obs = transform(obs).unsqueeze(0)
                obs = obs.flatten().detach().numpy()

                action = net.activate(obs)
                action[0] = action[0] * 2 - 1  # steering
                obs, reward, done, _, _ = env.step(action)  # 行動を起こす

                cumulative += reward

                if reward < 0:
                    neg_reward += 1
                else:
                    neg_reward = 0

                if time_count > time_limit or neg_reward > 20:
                    done = True

                if done:
                    env.close()
                    total_reward += cumulative
                    break

        genome.fitness = total_reward / 5
        print(f"id:{genome_id}, fitness: {genome.fitness}")


if __name__ == '__main__':
    # Load configuration.
    conf = Config(DefaultGenome, DefaultReproduction,
                  DefaultSpeciesSet, DefaultStagnation, './config/single.cfg')

    # Create population
    p = Population(conf)

    # run
    winner = p.run(eval_genomes, 300)

    # make result directory
    result_dir = './result_single' + str(int(time.time()))
    os.mkdir(result_dir)

    winner = p.reporters.best_genome()
    # save best genome
    np.save(result_dir + '/best.npy', winner)

    # print best genome
    print(winner)
