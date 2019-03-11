# -*- coding: utf-8 -*-

import gym
import gym.spaces
import gym_foodhunting

import pybullet as p

def getAction():
    keys = p.getKeyboardEvents()
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        return 0
    elif p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        return 1
    elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        return 2
    else:
        return 0

def main():
    env = gym.make('FoodHuntingHSRDiscreteGUI-v7')
    # env = gym.make('FoodHuntingDiscrete-v0')
    print(env.observation_space, env.action_space)
    obs = env.reset()
    while True:
        action = getAction()
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(action, obs, reward, done, info)
        if done:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    main()
