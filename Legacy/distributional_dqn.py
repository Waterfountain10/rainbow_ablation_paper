# also known as categorical DQN
# to do by max
from typing import Dict, Tuple
from matplotlib.axis import Ticker
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import sys

import gymnasium as gym
import gym_anytrading
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import torch.nn.functional as F
from tqdm import tqdm
from util.NeuralNet import NeuralNet
from util.RewardsNeuralNet import RewardsNeuralNet
from util.running_mean import running_mean
from torch.nn.utils import clip_grad_norm_

from dqn import DQN


class DistributionalDQN(DQN):
    '''
    The idea behind distributional DQN is to learn a distribution over the rewards isntead of a single
    value. This is done by using a neural network to approximate the distribution of the Q-values, which
    is then passed through a softmax to obtain the probabilities of each distribution. The distribution
    is split over a number of atoms, which can be thought of as discrete bins for the Q-values.
    '''

    def __init__(
        self,
        env: gym.Env,
        mem_size: int,
        batch_size: int,
        target_update_freq: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.99,
        alpha: float = 1e-3,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
    ):
        """Init"""
        self.env = env
        super().__init__(env, mem_size, batch_size, target_update_freq,
                         epsilon_decay, max_epsilon, min_epsilon, gamma, alpha)
        assert (
            env.observation_space.shape is not None), "Observation space must be a vector"
        self.obs_shape = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
        else:
            raise ValueError("Action space must be discrete")

        # Distributional DQN fields
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        self.support = torch.linspace(v_min, v_max, atom_size).to(
            self.device)  # shape (atom_size,)
        self.dqn_network = RewardsNeuralNet(
            input_dim=self.obs_shape,
            output_dim=int(self.action_dim),
            atom_size=self.atom_size,
            support=self.support,
        ).to(self.device)
        self.dqn_target = RewardsNeuralNet(
            input_dim=self.obs_shape,
            output_dim=int(self.action_dim),
            atom_size=self.atom_size,
            support=self.support,
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())
        self.dqn_target.eval()

        # Change optimizer to RMSprop
        self.optimizer = torch.optim.RMSprop(
            self.dqn_network.parameters(),
            lr=1e-3,  # Use higher learning rate with RMSprop
            alpha=0.95
        )

        # Add distribution tracking
        self.distribution_history = []
        self.distribution_episodes = []
        self.tracking_interval = 50  # Track distribution every 50 episodes
        self.loss = []

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        '''
        Literally same thing as DQN, but with a different network.
        '''
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_flat = obs.flatten()
            obs_tensor = (
                torch.as_tensor(obs_flat, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                q_vaues = self.dqn_network(obs_tensor)
            return q_vaues.argmax(dim=1).item()

    def step(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.float32, bool]:
        """
        again unchanged from DQN
        """
        action = self.select_action(state)
        next_state, reward, terminated, trucated, _ = self.env.step(action)
        reward = float(reward)
        done = terminated or trucated

        self.memory.store(state, int(action), reward, next_state, done)
        self.total_steps += 1
        self.epsilon = max(self.min_epsilon, self.epsilon -
                           self.epsilon_decay_rate)

        return action, next_state, np.float32(reward), done

    def update_model(self) -> float:
        '''
        Unchanged from DQN, but with a different loss function.
        '''
        samples = self.memory.sample_batch()
        # Convert ReplayBufferReturn to a dictionary if necessary
        samples_dict = {
            "obs": samples["obs"],
            "next_obs": samples["next_obs"],
            "acts": samples["acts"],
            "rews": samples["rews"],
            "done": samples["done"],
        }
        loss = self._compute_dqn_loss(samples_dict)

        self.optimizer.zero_grad()
        loss.backward()
        # After loss.backward()
        total_norm = 0.0
        for p in self.dqn_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f'Grad = {total_norm}, Loss: {loss}')
        torch.nn.utils.clip_grad_norm_(self.dqn_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''
        Distributional DQN uses cross entropy loss instead of MSE loss.
        The target distribution is computed using the Bellman equation,
        and is projected onto the support of the distribution.
        '''

        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(
            samples["rews"]).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(samples["done"]).unsqueeze(1).to(self.device)

        with torch.no_grad():
            q_next = self.dqn_target(next_state).argmax(1)
            dist_next, _ = self.dqn_target(next_state, return_prob=True)
            # shape (batch_size, atom_size)
            dist_next = dist_next[range(self.batch_size), q_next]

            # projection calculations
            Tz = reward + self.gamma * self.support * (1 - done)
            # ensure in range [v_min, v_max]
            Tz = torch.clamp(Tz, self.v_min, self.v_max)
            b = (Tz - self.v_min) / self.delta_z  # scale to [0, atom_size - 1]
            l = b.floor().long()  # lower bound
            u = b.ceil().long()  # upper bound

            # offset for indexing in flattened tensor
            offset = torch.linspace(0, (self.batch_size - 1) * self.atom_size,
                                    self.batch_size).long().unsqueeze(1).to(self.device)

            # init projection tensor
            proj = torch.zeros(
                (self.batch_size, self.atom_size)).to(self.device)
            # Distribute probability mass from next_prob into projection (lower & upper atoms)
            proj.view(-1).index_add_(
                0, (l + offset).view(-1), (dist_next * (u.float() - b)).view(-1)
            )
            proj.view(-1).index_add_(
                0, (u.clamp(max=self.atom_size - 1) +
                    offset).view(-1), (dist_next * (b - l.float())).view(-1)
            )

        # current distribution
        dist, _ = self.dqn_network(state, return_prob=True)

        # select the distribution for the action taken
        log_p = torch.log(dist[range(self.batch_size), action.squeeze()])
        # cross entropy loss
        loss = -(proj * log_p).sum(1).mean()

        return loss

    def _target_hard_update(self):
        '''Every target_update_freq steps, target_net <- copy(current_net)'''
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

    def train(self, num_episodes, show_progress=True):
        rewards = []

        episode_bar = None
        if show_progress:
            episode_bar = tqdm(total=num_episodes,
                               desc="Episodes", leave=False)

        # Use fixed state for consistent distribution comparison
        fixed_state, _ = self.env.reset()

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0
            steps_n = 0

            while not done:
                action, next_state, reward, done = self.step(state)

                # only update if batch has enough samples
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    self.loss.append(loss)

                state = next_state
                ep_reward += reward
                steps_n += 1

            # update target network if needed
            if episode % self.target_update_freq == 0:
                self._target_hard_update()

            rewards.append(ep_reward)
            if show_progress and episode_bar is not None:
                episode_bar.update(1)
                episode_bar.set_postfix(
                    reward=f"{ep_reward:.1f}", steps=steps_n)

            if episode % 20 == 0:  # Debug prints
              with torch.no_grad():
                  sample_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                  q_values = self.dqn_network(sample_state)
                  print(f"Episode {episode}, Avg Q-value: {q_values.mean().item():.3f}")
                  print(f"Epsilon: {self.epsilon:.3f}, Loss: {loss if 'loss' in locals() else 'N/A'}")

            if episode % 500 == 0 or episode == num_episodes - 1:  # Visualize less frequently
                self.compare_distributions(state, action, episode)

            # Track distributions periodically using the fixed state
            if episode % self.tracking_interval == 0:
                self.track_distribution(fixed_state, episode)

        if show_progress and episode_bar is not None:
            episode_bar.close()
        self.env.close()

        # After training, plot the evolution of distributions
        self.plot_distribution_evolution()
        self.plot_loss()

        return rewards

    def compare_distributions(self, state, action, episode_num):
        """Compare current and target distributions for debugging"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state, _, _, _, _ = self.env.step(action)  # Get next state
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([1.0]).unsqueeze(0).to(self.device)  # Assuming reward=1 for CartPole
            done = torch.FloatTensor([0.0]).unsqueeze(0).to(self.device)  # Assuming not done

            # Get current distribution
            current_dist, _ = self.dqn_network(state_tensor, return_prob=True)
            current_dist = current_dist[0, action].cpu().numpy()

            # Calculate target distribution (simplified version of what happens in _compute_dqn_loss)
            q_next = self.dqn_target(next_state_tensor).argmax(1)
            dist_next, _ = self.dqn_target(next_state_tensor, return_prob=True)
            dist_next = dist_next[0, q_next[0]].cpu().numpy()

            # Plot both distributions
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(self.support.cpu().numpy(), current_dist, alpha=0.7)
            plt.title(f"Current Distribution (Action {action})")
            plt.xlabel("Return Value")
            plt.ylabel("Probability")

            plt.subplot(1, 2, 2)
            plt.bar(self.support.cpu().numpy(), dist_next, alpha=0.7, color='orange')
            plt.title(f"Next State Distribution (Best Action)")
            plt.xlabel("Return Value")
            plt.ylabel("Probability")

            plt.tight_layout()
            plt.show()

    def track_distribution(self, state, episode_num):
        """Store distribution data for a fixed state at specific episode"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get distributions for all actions
            dist, q_values = self.dqn_network(state_tensor, return_prob=True)

            # Get best action's distribution
            best_action = q_values[0].argmax().item()
            action_dist = dist[0, best_action].cpu().numpy()

            # Store the distribution and episode number
            self.distribution_history.append(action_dist)
            self.distribution_episodes.append(episode_num)

            # Print a simple tracking message
            print(f"Tracked distribution at episode {episode_num}, Q-value: {q_values[0, best_action].item():.2f}")

    def plot_distribution_evolution(self):
        """Create a comprehensive visualization of how distributions evolved during training"""
        # Skip if we don't have enough data
        if len(self.distribution_history) < 2:
            return

        # Create a 2D visualization showing the evolution of distributions
        plt.figure(figsize=(12, 8))

        # Plot as a heatmap
        num_distributions = len(self.distribution_history)
        support = self.support.cpu().numpy()

        # Create a 2D array for the heatmap
        dist_array = np.array(self.distribution_history)

        # Plot heatmap
        plt.imshow(dist_array,
                  aspect='auto',
                  extent=[support[0], support[-1], self.distribution_episodes[-1], self.distribution_episodes[0]],
                  cmap='viridis',
                  interpolation='nearest')

        plt.colorbar(label='Probability')
        plt.xlabel('Return Value')
        plt.ylabel('Episode')
        plt.title('Evolution of Value Distribution During Training')

        # Also create a 3D visualization showing select distributions
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot a subset of distributions (to avoid overcrowding)
        max_to_show = min(10, num_distributions)
        indices = np.linspace(0, num_distributions-1, max_to_show, dtype=int)

        # Create x, y coordinates for the 3D plot
        x = np.arange(len(support))
        for i in indices:
            y = self.distribution_episodes[i]
            z = self.distribution_history[i]
            ax.bar(x, z, zs=y, zdir='y', alpha=0.8, width=0.8)

        ax.set_xlabel('Atom Index')
        ax.set_ylabel('Episode')
        ax.set_zlabel('Probability')
        ax.set_title('Value Distributions During Training')

        # Set x-ticks to show atom values
        step = max(1, len(support) // 10)  # Show at most 10 ticks
        ax.set_xticks(np.arange(0, len(support), step))
        ax.set_xticklabels([f"{v:.1f}" for v in support[::step]])

        # Save figures
        os.makedirs("distribution_plots", exist_ok=True)
        plt.savefig(f"distribution_plots/distribution_evolution.png")
        plt.show()

    def plot_loss(self):
        """Plot the loss over episodes"""
        if len(self.loss) < 2:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.loss, label='Loss')
        plt.title('Loss Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Save figure
        os.makedirs("distribution_plots", exist_ok=True)
        plt.savefig(f"distribution_plots/loss_over_episodes.png")
        plt.show()


if __name__ == "__main__":
   # Parameters for DQN
    MEMORY_SIZE = 150
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10
    EPSILON_DECAY_STEPS = 5000
    LEARNING_RATE = 1e-5  # Try much smaller (current is 5e-4)
    # Small number for testing (increased it to compare with PER - will)
    NUM_TOTAL_EPISODES = 2000
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    # env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=50)

    agent = DistributionalDQN(
        env=env,
        mem_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        epsilon_decay=EPSILON_DECAY_STEPS,
        alpha=LEARNING_RATE,
        min_epsilon=0.01,
        # Categorical DQN parameters
        v_min=-50,
        v_max=400,
        atom_size=51,
    )

    rewards = agent.train(NUM_TOTAL_EPISODES)
    # print("Rewards at end:", np.mean(rewards))

    # PLOT GRAPH AND SAVE IT
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="episode Reward", alpha=0.6)

    if len(rewards) >= 10:  # apply cumsum sliding mean
        smoothed = running_mean(rewards, window_size=10)
        plt.plot(
            range(10 - 1, len(rewards)), smoothed, label="smoothed window 10", linewidth=2
        )
    plt.title("DQN training rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # also save png SAVE DID NOT WORK BTW
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/rewards_DistributionalDQN.png")
    print("Plot saved to results/rewards_DistributionalDQN.png")

    plt.show()
