# PyLIS

PyLIS (Life in Silico with PyBullet) is a 3D simulation environment for research in human level artificial intelligence (HLAI) and artificial general intelligence (AGI). PyLIS includes agent (robot) models with wheels for locomotion and a manipulator; more specifically:

- Software library for:
	- reading robot models into the virtual 3D space with a physics engine
	- obtaining color images and depth images from the head camera
	- moving by wheels
	- manipulating objects with a manipulator
	- detecting collision with other objects
- A sample environment and agents to perform reinforcement learning with the library.

PyLIS can be used in research involving both manipulation and locomotion capabilities. The combination of manipulation and locomotion in tasks makes them much more complex than cases where those capabilities are tested separately (because of its expanded search space).

PyLIS can also be used for research in social interaction with object manipulation/operation with more than one agent, including research in acquisition of language involving object expressions.

See [wiki](https://github.com/wbap/PyLIS/wiki) for more details.

![screenshot](https://raw.githubusercontent.com/wbap/PyLIS/master/misc/pylis.png)

# Demo Videos

Click on thumbnail to play youtube video.

## Manipulator Demo

[![PyLIS manipulator demo 1](https://img.youtube.com/vi/5H6bFS57Uqw/0.jpg)](https://www.youtube.com/watch?v=5H6bFS57Uqw)

## Locomotion Demo

[![PyLIS locomotion demo 1](https://img.youtube.com/vi/kB8RLlHuNUE/0.jpg)](https://www.youtube.com/watch?v=kB8RLlHuNUE)

## Reinforcement Learning Demo

[![PyLIS reinforcement learning demo 1](https://img.youtube.com/vi/OwSosoGb16A/0.jpg)](https://www.youtube.com/watch?v=OwSosoGb16A)


# Available Environments

## Food Hunting

- https://github.com/wbap/PyLIS/tree/master/gym-foodhunting


# Links

WBAI's Request for Reseach: 3D Agent Test Suites

- https://wba-initiative.org/en/research/rfr/3d-agent-test-suites/

OpenAI Gym

- https://gym.openai.com/
- https://github.com/openai/gym

Bullet Real-Time Physics Simulation

- https://pybullet.org/
- https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
- https://github.com/bulletphysics/bullet3

Stable Baselines

- https://github.com/hill-a/stable-baselines
- https://stable-baselines.readthedocs.io/


# Robot models

This repository supports these robot models.

## Toyota HSR by TRI

URDF robot models for Toyota HSR by TRI.

- https://github.com/ToyotaResearchInstitute/hsr_description

3D Mesh files for Toyota HSR by TRI.

- https://github.com/ToyotaResearchInstitute/hsr_meshes

## R2D2 by Bullet Physics

Bullet Physics SDK.

- https://github.com/bulletphysics/bullet3


# Acknowledgment

This work was supported by MEXT KAKENHI Grant Number 17H06308, Grant-in-Aid for Scientific Research on Innovative Areas, Brain information dynamics underlying multi-area interconnectivity and parallel processing.

- http://brainfodynamics.umin.jp/


# Author

Susumu OTA
