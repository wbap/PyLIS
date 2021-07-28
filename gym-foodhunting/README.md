# gym-foodhunting

Gym environments and agents for food hunting in the 3D world.


# Install

I've tested on Mac OS X 10.13.6 (Python 3.7.5) and Ubuntu 16.04.

Some packages need to install prerequisites. See these pages for more details.

- https://www.scipy.org/install.html
- https://github.com/openai/gym#installation
- https://opencv.org/
- https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
- https://github.com/DLR-RM/stable-baselines3

```
git clone https://github.com/wbap/PyLIS.git
cd PyLIS

python3 -m venv venv
source venv/bin/activate

pip install numpy

pip install gym

pip install opencv-python

pip install pybullet

# You may need to install prerequisites.
# See https://github.com/DLR-RM/stable-baselines3#installation for more details.
pip install stable-baselines3

git clone https://github.com/ToyotaResearchInstitute/hsr_meshes.git
cp -rp hsr_meshes venv/lib/python3.7/site-packages/pybullet_data

cp -p gym-foodhunting/urdf/hsrb4s.urdf venv/lib/python3.7/site-packages/pybullet_data
# If you want to use original URDF file by TRI,
# git clone https://github.com/ToyotaResearchInstitute/hsr_description.git
# cp -p hsr_description/robots/hsrb4s.urdf venv/lib/python3.7/site-packages/pybullet_data
cp -p gym-foodhunting/urdf/r2d2.urdf venv/lib/python3.7/site-packages/pybullet_data
cp -p gym-foodhunting/urdf/food_sphere.urdf venv/lib/python3.7/site-packages/pybullet_data
cp -p gym-foodhunting/urdf/food_cube.urdf venv/lib/python3.7/site-packages/pybullet_data

cd gym-foodhunting
pip install -e .
cd ..
```


# Uninstall

```
pip uninstall gym-foodhunting

pip uninstall stable-baselines3
pip uninstall opencv-python
pip uninstall pybullet
pip uninstall gym
pip uninstall numpy

# or just remove venv directory.
```


# Example

## Simplest example

```python
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
    env = gym.make('FoodHuntingDiscreteGUI-v0')
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
```

## Simple reinforcement learning example

```python
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
            print(info)
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
```

```
# Learn
python examples/example_rl.py --env_name="FoodHuntingDiscrete-v0" --total_timesteps=10000 --filename="saved_model"
```

```
# Play without GUI
python examples/example_rl.py --env_name="FoodHuntingDiscrete-v0" --total_timesteps=10000 --filename="saved_model" --play

# Play with GUI
python examples/example_rl.py --env_name="FoodHuntingDiscreteGUI-v0" --total_timesteps=10000 --filename="saved_model" --play
```

## More practical reinforcement learning example

See https://github.com/wbap/PyLIS/blob/master/gym-foodhunting/agents/ppo_agent.py

```
cd gym-foodhunting

# Run this to enable SubprocVecEnv on Mac OS X.
# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# see https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331

# See available options.
python agents/ppo_agent.py --help

# Learn
# This may take a few hours.
time python agents/ppo_agent.py --env_name="FoodHuntingHSRDiscrete-v1" --total_timesteps=500000 --n_cpu=16 --tensorboard_log="tblog"

# Monitor
tensorboard --logdir tblog
# Open web browser and access http://localhost:6006/

# Play with GUI
# This will open PyBullet window.
time python agents/ppo_agent.py --env_name="FoodHuntingHSRDiscrete-v1" --load_file="FoodHuntingHSR-v1_best.pkl" --total_timesteps=500000 --n_cpu=16 --play
```


# Available Environments

```

# R2D2 model
FoodHunting-v0             # continuous action, without rendering
FoodHuntingGUI-v0          # continuous action, with rendering
FoodHuntingDiscrete-v0     # discrete action, without rendering
FoodHuntingDiscreteGUI-v0  # discrete action, with rendering

# HSR model
FoodHuntingHSR-v0
FoodHuntingHSRGUI-v0
FoodHuntingHSR-v1
FoodHuntingHSRGUI-v1
FoodHuntingHSRDiscrete-v0
FoodHuntingHSRDiscreteGUI-v0
FoodHuntingHSRDiscrete-v1
FoodHuntingHSRDiscreteGUI-v1
```

# Author

Susumu OTA
