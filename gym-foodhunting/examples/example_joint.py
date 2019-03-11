# -*- coding: utf-8 -*-

import numpy as np
import gym
import gym.spaces
import gym_foodhunting

import pybullet as p

# for HSR
paramNames = [
    'wheel left', 'wheel right',
    'base roll',
    'torso lift',
    'head pan', 'head tilt',
    'arm lift', 'arm flex', 'arm roll',
    'wrist flex', 'wrist roll',
    #'hand motor',
    #'hand left proximal', 'hand left spring proximal', 'hand left mimic distal', 'hand left distal',
    #'hand right proximal', 'hand right spring proximal', 'hand right mimic distal', 'hand right distal'
]

# for R2D2
# paramNames = [
#     'wheel left', 'wheel right',
#     'gripper extension', 'gripper left', 'gripper right',
#     'head pan'
# ]

def main():
    env = gym.make('FoodHuntingHSRGUI-v2')
    # env = gym.make('FoodHuntingHSRTestGUI-v0')
    params = [ p.addUserDebugParameter(paramName, -1.0, 1.0, 0.0) for paramName in paramNames ]
    obs = env.reset()
    while True:
        action = [ p.readUserDebugParameter(param) for param in params ]
        obs, reward, done, info = env.step(np.array(action))
        # print(action, obs, reward, done, info)
        if done:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    main()
