# -*- coding: utf-8 -*-

import argparse
import gym
import gym_foodhunting

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

def learn(env_name, save_file, total_timesteps):
    env = DummyVecEnv([lambda: gym.make(env_name)])
    model = PPO(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_file)
    del model
    env.close()

def play(env_name, load_file, total_timesteps):
    env = DummyVecEnv([lambda: gym.make(env_name)])
    model = PPO.load(load_file, verbose=1)
    obs = env.reset()
    for i in range(total_timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # env.render() # dummy
        if done:
            print(info[0]['episode'])
    del model
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='play or learn.')
    parser.add_argument('--env_name', type=str, default='FoodHuntingDiscreteGUI-v0', help='environment name.')
    parser.add_argument('--filename', type=str, default='saved_model', help='filename to save/load model.')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='total timesteps.')
    args = parser.parse_args()
    if args.play:
        play(args.env_name, args.filename, args.total_timesteps)
    else:
        learn(args.env_name, args.filename, args.total_timesteps)
