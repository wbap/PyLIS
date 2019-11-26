# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import gym
import gym_foodhunting

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

BEST_SAVE_FILE_SUFFIX = '_best'

def make_env(env_name, rank, seed):
    def _init():
        env = gym.make(env_name)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def learn(env_name, seed, load_file, save_file, tensorboard_log, total_timesteps, n_cpu):
    best_mean_reward = -np.inf
    best_mean_step = np.inf
    save_file = env_name if save_file is None else save_file
    best_save_file = save_file + BEST_SAVE_FILE_SUFFIX
    start_time = time.time()
    def callback(_locals, _globals):
        nonlocal best_mean_reward, best_mean_step
        t = (time.time() - start_time) / 3600.0
        print(f'hours: {t:.2f}')
        ep_info_buf = _locals['ep_info_buf']
        if len(ep_info_buf) < ep_info_buf.maxlen:
            return True
        mean_reward = np.mean([ ep_info['r'] for ep_info in ep_info_buf ])
        mean_step = np.mean([ ep_info['l'] for ep_info in ep_info_buf ])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print('best_mean_reward:', best_mean_reward)
            print('saving new best model:', best_save_file)
            _locals['self'].save(best_save_file)
        if mean_step < best_mean_step:
            best_mean_step = mean_step
            print('best_mean_step:', best_mean_step)
        return True # False should finish learning
    # policy = CnnPolicy
    policy = CnnLstmPolicy
    # policy = CnnLnLstmPolicy
    print(env_name, policy)
    # Run this to enable SubprocVecEnv on Mac OS X.
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    # see https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331
    env = SubprocVecEnv([make_env(env_name, i, seed) for i in range(n_cpu)])
    if load_file is not None:
        model = PPO2.load(load_file, env, verbose=1, tensorboard_log=tensorboard_log)
    else:
        model = PPO2(policy, env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=total_timesteps, log_interval=5, callback=callback)
    print('saving model:', save_file)
    model.save(save_file)
    env.close()

def play(env_name, seed, load_file, total_timesteps, n_cpu):
    np.set_printoptions(precision=5)
    def padding_obss(obss, dummy_obss):
        dummy_obss[ 0, :, :, : ] = obss
        return dummy_obss
    # trained LSTM model cannot change number of env.
    # so it needs to reshape observation by padding dummy data.
    dummy_obss = np.zeros((n_cpu, 64, 64, 4))
    env = SubprocVecEnv([make_env(env_name, 0, seed)])
    model = PPO2.load(load_file, verbose=1)
    obss = env.reset()
    obss = padding_obss(obss, dummy_obss)
    rewards_buf = []
    steps_buf = []
    # TODO: single
    for i in range(total_timesteps):
        actions, _states = model.predict(obss)
        actions = actions[0:1]
        obss, rewards, dones, infos = env.step(actions)
        obss = padding_obss(obss, dummy_obss)
        # env.render() # dummy
        if dones[0]:
            rewards_buf.append(infos[0]['episode']['r'])
            steps_buf.append(infos[0]['episode']['l'])
            line = np.array([np.mean(rewards_buf), np.std(rewards_buf), np.mean(steps_buf), np.std(steps_buf)])
            print(len(rewards_buf), line)
            #obss = env.reset()
            #obss = padding_obss(obss, dummy_obss)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='play or learn.')
    parser.add_argument('--env_name', type=str, default='FoodHunting-v0', help='environment name.')
    parser.add_argument('--load_file', type=str, default=None, help='filename to load model.')
    parser.add_argument('--save_file', type=str, default=None, help='filename to save model.')
    parser.add_argument('--tensorboard_log', type=str, default=None, help='tensorboard log file.')
    parser.add_argument('--total_timesteps', type=int, default=10000000, help='total timesteps.')
    parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU cores.')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number.')
    args = parser.parse_args()

    if args.play:
        play(args.env_name, args.seed, args.load_file, args.total_timesteps, args.n_cpu)
    else:
        learn(args.env_name, args.seed, args.load_file, args.save_file, args.tensorboard_log, args.total_timesteps, args.n_cpu)
