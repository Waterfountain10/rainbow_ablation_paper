# --------------------------------------------------------------------------
# CLI to run any agent class:
#   > select agent via --agent
#   > specify env and seed
#   > logs results into results{agent_name}
#
# ex. python run.py --agent dueling --env Stocks-v0 --save_dir results/dueling
# --------------------------------------------------------------------------
# CLI to resume training
# ex. python run.py --agent rainbow --seed 1337 --resume
# --------------------------------------------------------------------------
import argparse
import gymnasium as gym
import torch
import numpy as np
from multiprocessing import Process

AGENTS = [
    "DQN",
    "DDQN",
    "Prioritized DDQN",
    "Dueling DDQN",
    "Noisy DQN",
    "Distributional DQN",
    "Multi-Step Learning DQN",
    "Rainbow DQN",
]
BASE_SEED = 69

# Multi Processing: to speed up batch learning, we ware training with 4+ processes
def make_env(env_name, seed):
    def _thunk(): # make a funciton thunk (or delayed computation)
        import torch
        torch.set_num_threads(1) # avoid over-using CPU threads
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env
    return _thunk

def make_parallel_env(env_name, num_envs, seed):
    # each process needs a different seed to avoid duplicating results
    envs = []
    for i in range(num_envs):
        p = Process(target=make_env, args=(env_name, seed+i)) # processes will have seed = i, i+1, i+2...
        p.run()
        envs.append(p)

    return envs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True, help='Agent name (e.g., dqn, dueling, rainbow)')
    parser.add_argument('--env', type=str, default='Stocks-v0', help='Gym environment ID')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='Total steps to train')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    envs = make_parallel_env(args.env, args.num_envs, seed=BASE_SEED)
    print(envs)

    # TO DO:
    #  load agent from --agent, --env (just one here), and train that bad boy
    #  save results in /results in .npy
    #  dont forget to put tqdm


if __name__ == "__main__":
    main()
