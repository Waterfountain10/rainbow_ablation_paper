import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from params import NUMBER_STEPS, WINDOW_SIZE, parse_args, NUMBER_TEST_EPISODES
from util.running_mean import running_mean
from LoadData import load_dataset
import gym_anytrading
from combined_agent import CombinedAgent
import os

'''
===========================================================================
                  Importing Hyperparameters from params.py
===========================================================================
'''
args = parse_args()

MEMORY_SIZE = args.memory_size
BATCH_SIZE = args.batch_size
TARGET_UPDATE_FREQ = args.target_update_freq

EPSILON_DECAY_STEPS = int(args.epsilon_decay_steps)
LEARNING_RATE = args.lr
NUM_TOTAL_EPISODES = args.num_episodes
MIN_EPSILON = args.min_epsilon

default_params = {
    "omega": args.omega,
    "beta": args.beta,
    "td_epsilon": args.td_epsilon,
    "v_min": args.v_min,
    "v_max": args.v_max,
    "atom_size": args.atom_size,
    "n_step": args.n_step,
    "sigma_init": args.sigma_init,
    "gamma": args.gamma,
}

'''
===========================================================================
                  Setting Configuration for Combined Agent
===========================================================================
'''
rainbow_config = {
    "useDouble": args.useDouble,
    "usePrioritized": args.usePrioritized,
    "useDuel": args.useDuel,
    "useNoisy": args.useNoisy,
    "useDistributive": args.useDistributive,
    "useNstep": args.useNstep,
}

print(rainbow_config)

'''
===========================================================================
                  Loading Data and Creating Environments
===========================================================================
'''
# creating the env
env = gym.make(args.env, max_episode_steps=NUMBER_STEPS)

'''
===========================================================================
                            Creating Agent
===========================================================================
'''
agent = CombinedAgent(
    env=env,
    mem_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    epsilon_decay=EPSILON_DECAY_STEPS,
    alpha=LEARNING_RATE,
    min_epsilon=MIN_EPSILON,
    agent_config=rainbow_config,
    combined_params=default_params,
    gamma=default_params["gamma"],
    hidden_dim = int(args.hidden_dim)
)

'''
===========================================================================
                  Training Agent and Saving Results
===========================================================================
'''
# Create checkpoints directory if it doesn't exist
os.makedirs("test_checkpoints", exist_ok=True)

# Setting model name for saving
config_components = [k[3:] for k, v in rainbow_config.items() if v]
model_name = "_".join(config_components)
params = f"mem{MEMORY_SIZE:d}_batch{BATCH_SIZE:d}_lr{LEARNING_RATE}_minEps{MIN_EPSILON}_gamma{args.gamma}_hiddenDim{args.hidden_dim}"

# TODO: change for training
checkpoint_filename = f"test_checkpoints/{model_name}" +".npy"

if os.path.exists(checkpoint_filename):
    # Skip training if the file already exists
    print(f"File with name {model_name}.npy already exists. Skipping training.")
else:
    print("=============================================================")
    print(f"Beginning training {model_name}.npy")
    print("=============================================================")
    # Train and save the return values
    rewards = agent.train(NUM_TOTAL_EPISODES)
    np.save(checkpoint_filename, rewards)
    print(f"Training complete. Rewards saved to {checkpoint_filename}")
