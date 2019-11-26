from gym.envs.registration import register
from gym_foodhunting.foodhunting.gym_foodhunting import R2D2, R2D2Simple, R2D2Discrete, HSR, HSRSimple, HSRDiscrete

# FoodHunting R2D2
register(
    id='FoodHunting-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': False, 'robot_model': R2D2Simple, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': True, 'robot_model': R2D2Simple, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': False, 'robot_model': R2D2Discrete, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': True, 'robot_model': R2D2Discrete, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

# FoodHunting HSR
register(
    id='FoodHuntingHSR-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': False, 'robot_model': HSRSimple, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': True, 'robot_model': HSRSimple, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSR-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': False, 'robot_model': HSRSimple, 'max_steps': 50, 'num_foods': 1, 'num_fakes': 1, 'object_size': 0.5, 'object_radius_scale': 0.0, 'object_radius_offset': 1.5, 'object_angle_scale': 0.25}
)

register(
    id='FoodHuntingHSRGUI-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': True, 'robot_model': HSRSimple, 'max_steps': 50, 'num_foods': 1, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 0.0, 'object_radius_offset': 1.5, 'object_angle_scale': 0.25}
)

register(
    id='FoodHuntingHSRDiscrete-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': False, 'robot_model': HSRDiscrete, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRDiscreteGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': True, 'robot_model': HSRDiscrete, 'max_steps': 50, 'num_foods': 2, 'num_fakes': 0, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRDiscrete-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': False, 'robot_model': HSRDiscrete, 'max_steps': 50, 'num_foods': 1, 'num_fakes': 1, 'object_size': 0.5, 'object_radius_scale': 0.0, 'object_radius_offset': 1.5, 'object_angle_scale': 0.25}
)

register(
    id='FoodHuntingHSRDiscreteGUI-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=50,
    kwargs={'render': True, 'robot_model': HSRDiscrete, 'max_steps': 50, 'num_foods': 1, 'num_fakes': 1, 'object_size': 0.5, 'object_radius_scale': 0.0, 'object_radius_offset': 1.5, 'object_angle_scale': 0.25}
)

register(
    id='FoodHuntingHSRTestGUI-v0',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=10000,
    kwargs={'render': True, 'robot_model': HSR, 'max_steps': 10000, 'num_foods': 5, 'num_fakes': 5, 'object_size': 0.5, 'object_radius_scale': 1.0, 'object_radius_offset': 1.0, 'object_angle_scale': 1.0}
)

register(
    id='FoodHuntingHSRTestGUI-v1',
    entry_point='gym_foodhunting.foodhunting:FoodHuntingEnv',
    max_episode_steps=10000,
    kwargs={'render': True, 'robot_model': HSR, 'max_steps': 10000, 'num_foods': 1, 'num_fakes': 1, 'object_size': 0.5, 'object_radius_scale': 0.0, 'object_radius_offset': 1.5, 'object_angle_scale': 0.25}
)
