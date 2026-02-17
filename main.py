import gymnasium as gym
from environment_config import DefaultConfig, RandomMap

FROZEN_LAKE = "FrozenLake-v1"
CONFIG = RandomMap()
# This way we can easlily change our config

env = gym.make(
    FROZEN_LAKE,
    desc=CONFIG.desc,
    # customs map e.g desc=["SFFF", "FHFH", "FFFH", "HFFG"]
    # “S” for Start tile
    # “G” for Goal tile
    # “F” for frozen tile
    # “H” for a tile with a hole
    map_name=CONFIG.map_name,
    is_slippery=CONFIG.is_slippery,
    success_rate=CONFIG.success_rate,
    reward_schedule=CONFIG.reward_schedule,
    render_mode="human",  # this is not present in the config for debug purposes
)

observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"Starting observation: {observation}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = (
        env.action_space.sample()
    )  # Random action for now - real agents will be smarter!

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
