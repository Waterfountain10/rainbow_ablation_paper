from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random

import gymnasium as gym

from util.CombinedBuffer import CombinedBuffer
from util.CombinedNetwork import CombinedNeuralNet
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn.functional as F
from tqdm import tqdm
import gym_anytrading

from params import NUMBER_STEPS, NUMBER_TEST_EPISODES


class CombinedAgent:
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
        agent_config={
            "useDouble": False,
            "usePrioritized": False,
            "useDuel": False,
            "useNoisy": False,
            "useNstep": False,
            "useDistributive": False,
        },
        combined_params={
            # PER specific parameters:
            "omega": 0.6,  # priority importance parameter # TODO 0 to 1
            "beta": 0.4,  # then gets increased more later # TODO how it decays and 0.4 to 0.7
            "td_epsilon": 1e-6,
            # Categorical DQN parameters
            "v_min": 0.0,
            "v_max": 200.0,
            "atom_size": 51,
            # Nstep parameters
            "n_step": 3,  # TODO  0 to anything
            "sigma_init": 0.5,
        },
        hidden_dim: int = 256
    ):
        """Init"""
        self.env = env

        self.obs_shape = self.env.observation_space.shape
        assert self.obs_shape is not None
        self.gamma = gamma

        if not agent_config["useNoisy"]:
            self.epsilon = max_epsilon
            self.epsilon_decay = epsilon_decay
            self.max_epsilon = max_epsilon
            self.min_epsilon = min_epsilon
            # for linear
            self.epsilon_decay_rate = (
                (max_epsilon - min_epsilon) / epsilon_decay if epsilon_decay > 0 else 0
            )
            # for exponential decay rate: max * (decayRate)^eps_decay = min
            self.eps_exp_decay_rate = (min_epsilon / max_epsilon) ** (
                1.0 / epsilon_decay
            )

        self.agent_config = agent_config

        self.device = "cpu"

        # comment/uncomment below to use cpu/gpu
        if torch.cuda.is_available():
            self.device = "cuda"
        if torch.mps.is_available():
            self.device = "mps"
            
        print(self.device)

        if agent_config["usePrioritized"]:
            self.omega = combined_params["omega"]
            self.beta = combined_params["beta"]
            self.td_epsilon = combined_params["td_epsilon"]

        if agent_config["useDistributive"]:
            # Distributional DQN fields
            self.v_min = combined_params["v_min"]
            self.v_max = combined_params["v_max"]
            self.atom_size = combined_params["atom_size"]
            self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)
            self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
                self.device
            )  # shape (atom_size,)
            combined_params["support"] = self.support
            # Add distribution tracking
            self.distribution_history = []
            self.distribution_episodes = []
            self.tracking_interval = 50  # Track distribution every 50 episodes

        # Create n-step memory if needed
        if agent_config["useNstep"]:
            self.n_step = combined_params["n_step"]

            # Create n-step buffer with n-step enabled
            n_step_config = agent_config.copy()
            self.n_memory = CombinedBuffer(
                self.obs_shape,
                mem_size,
                batch_size=batch_size,
                buffer_config=n_step_config,
                combined_params=combined_params,
            )

            # Create regular buffer with n-step disabled
            regular_config = agent_config.copy()
            regular_config["useNstep"] = False
            self.memory = CombinedBuffer(
                self.obs_shape,
                mem_size,
                batch_size=batch_size,
                buffer_config=regular_config,
                combined_params=combined_params,
            )
        else:
            # Create standard buffer
            self.memory = CombinedBuffer(
                self.obs_shape,
                mem_size,
                batch_size=batch_size,
                buffer_config=agent_config,
                combined_params=combined_params,
            )

        # self.state_size = env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
        else:
            raise ValueError("Action space must be discrete")

        # Network should change depending on the agent_config
        self.dqn_network = CombinedNeuralNet(
            self.obs_shape,
            int(self.action_dim),
            network_config=self.agent_config,
            combined_params=combined_params,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.dqn_target = CombinedNeuralNet(
            self.obs_shape,
            int(self.action_dim),
            network_config=self.agent_config,
            combined_params=combined_params,
            hidden_dim=hidden_dim
        ).to(self.device)
        # make identical copies of the neural net
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

        self.dqn_target.train(False)

        # if self.agent_config["usePrioritized"]:
        #     # Change optimizer to RMSprop
        #     self.optimizer = torch.optim.RMSprop(
        #         self.dqn_network.parameters(),
        #         lr=1e-3,  # Use higher learning rate with RMSprop
        #         alpha=0.95,
        #     )
        # else:
        self.optimizer = torch.optim.Adam(self.dqn_network.parameters(), lr=alpha)

        self.batch_size = batch_size
        self.testing = False
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        self.updating_eps = True

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        if not self.agent_config["useNoisy"] and np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        obs_flat = obs.flatten()
        obs_tensor = (
            torch.as_tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_vaues = self.dqn_network(obs_tensor)
        return q_vaues.argmax().item()

    def step(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float32, bool]:
        """
        Go from current state -> next_state (and return everything related to this transition)
        Returns:
            action, next_state, reward, done
        """
        action = self.select_action(state)
        next_state, reward, terminated, trucated, _ = self.env.step(action)
        reward = float(reward)
        done = terminated or trucated
        self.total_steps += 1

        if self.agent_config["useNstep"]:
            have_enough_in_buffer = self.n_memory.store(
                state=state,
                action=int(action),
                reward=reward,
                next_state=next_state,
                done=done,
            )
            if have_enough_in_buffer:
                # need to add here bc in train it uses len(self.memory) and don't want to override train()
                self.memory.store(state, int(action), reward, next_state, done)
        else:
            self.memory.store(state, int(action), reward, next_state, done)

        # linear decay
        # self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay_rate)

        # exp decay
        if not self.agent_config["useNoisy"]:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_exp_decay_rate)

            if self.epsilon == self.min_epsilon and self.updating_eps:
                self.updating_eps = False
                print("epsilon at minimum")

        return action, next_state, np.float32(reward), done

    def update_model(self) -> torch.Tensor | float:
        if self.agent_config["useNstep"]:
            samples = self.n_memory.sample_batch()
            original_gamma = self.gamma
            self.gamma = self.gamma**self.n_step
        else:
            samples = self.memory.sample_batch()

        per_sample_losses, loss_for_backprop = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss_for_backprop.backward()

        # Apply appropriate gradient clipping
        clip_value = 10.0 if self.agent_config["useDistributive"] else 100.0
        torch.nn.utils.clip_grad_norm_(
            self.dqn_network.parameters(), max_norm=clip_value
        )

        self.optimizer.step()

        if self.agent_config["usePrioritized"]:
            # update priorities given our index array
            idxs = samples["idxs"]  # simple array type
            # untrack the gradients since this is not used for loss calculation but just priority tracking
            td_tensor = per_sample_losses.detach().cpu().numpy()
            td_tensor = td_tensor.squeeze()

            new_priorities = np.atleast_1d(
                abs(td_tensor + self.td_epsilon)
            )  # p_i = |delta_i| + epsilon
            idxs = np.array(idxs)
            # updates in buffer with : p_i ^ omega
            self.memory.update_priorities(idxs, new_priorities)

        if self.agent_config["useNoisy"]:
            # refresh epsilons in noisy layers inside our NoisyNet
            self.dqn_network.reset_noise()
            self.dqn_target.reset_noise()

        if self.agent_config["useNstep"]:
            # set gamma back to what it is supposed to be
            self.gamma = original_gamma  # type: ignore

        return (
            loss_for_backprop.item()
            if isinstance(loss_for_backprop, torch.Tensor)
            else loss_for_backprop
        )

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]):
        tensors = self._prepare_samples(samples)

        if self.agent_config["useDistributive"]:
            losses = self._compute_dqn_loss_distributive(tensors)  # type: ignore
        else:
            losses = self._compute_standard_dqn_loss(tensors)  # type: ignore

        if self.agent_config["usePrioritized"] and tensors["weights"] is not None:
            # calculate weighted loss rather than simple loss
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(
                self.device
            )
            # tensor with a single scalar
            weighted_loss = torch.mean(losses * weights)
            return losses, weighted_loss

        return losses, losses.mean() if losses.dim() > 0 else losses

    def _get_target_q_values(self, next_state):
        """Calculate target Q-values using either Double DQN or standard DQN."""
        if self.agent_config["useDouble"]:
            return self._get_double_dqn_target(next_state)
        else:
            return self._get_standard_dqn_target(next_state)

    def _get_double_dqn_target(self, next_state):
        """Calculate target Q-values using Double DQN method."""
        # Use online network to select actions
        q_next = self.dqn_network(next_state).argmax(1)
        return self.dqn_target(next_state).gather(1, q_next.unsqueeze(1))

    def _get_standard_dqn_target(self, next_state):
        """Calculate target Q-values using standard DQN method."""
        next_q_values = self.dqn_target(next_state)
        return next_q_values.max(dim=1, keepdim=True)[0]

    def _prepare_samples(
        self, samples: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor | np.ndarray | None]:
        """Convert sample dictionary to device tensors."""
        return {
            "state": torch.FloatTensor(samples["obs"]).to(self.device),
            "next_state": torch.FloatTensor(samples["next_obs"]).to(self.device),
            "action": torch.LongTensor(samples["acts"]).unsqueeze(1).to(self.device),
            "reward": torch.FloatTensor(samples["rews"]).unsqueeze(1).to(self.device),
            "done": torch.FloatTensor(samples["done"]).unsqueeze(1).to(self.device),
            "idxs": samples.get("idxs"),
            "weights": samples.get("weights"),
        }

    def _compute_standard_dqn_loss(
        self,
        samples: Dict[str, torch.Tensor | np.ndarray],
    ) -> torch.Tensor:
        # Get current Q-values
        q_values = self.dqn_network(samples["state"])
        q_current = q_values.gather(1, samples["action"])

        # Get target Q-values
        with torch.no_grad():
            q_next = self._get_target_q_values(samples["next_state"])
            q_target = samples["reward"] + self.gamma * q_next * (1 - samples["done"])

        return F.smooth_l1_loss(q_current, q_target, reduction="none")

    def _compute_dqn_loss_distributive(
        self,
        samples: Dict[str, torch.Tensor | np.ndarray],
    ) -> torch.Tensor:
        """Compute the loss for distributional DQN using cross entropy and projecting onto nearest atoms."""
        with torch.no_grad():
            if self.agent_config["useDouble"]:
                # Use online network to select actions
                q_next = self.dqn_network(samples["next_state"]).argmax(1)
            else:
                # Use target network to select actions
                q_next = self.dqn_target(samples["next_state"]).argmax(1)
            dist_next, _ = self.dqn_target(samples["next_state"], return_prob=True)
            # shape (batch_size, atom_size)
            dist_next = dist_next[range(self.batch_size), q_next]

            # projection calculations
            Tz = samples["reward"] + self.gamma * self.support * (1 - samples["done"])
            # print(torch.max(Tz))
            # ensure in range [v_min, v_max]
            # type: ignore since rewards and done are Tensors
            Tz = torch.clamp(Tz, self.v_min, self.v_max)  # type: ignore
            b = (Tz - self.v_min) / self.delta_z  # scale to [0, atom_size - 1]
            l = b.floor().long()  # lower bound
            u = b.ceil().long()  # upper bound

            # offset for indexing in flattened tensor
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .to(self.device)
            )

            # init projection tensor
            proj = torch.zeros((self.batch_size, self.atom_size)).to(self.device)
            # Distribute probability mass from next_prob into projection (lower & upper atoms)
            proj.view(-1).index_add_(
                0, (l + offset).view(-1), (dist_next * (u.float() - b)).view(-1)
            )
            proj.view(-1).index_add_(
                0,
                (u.clamp(max=self.atom_size - 1) + offset).view(-1),
                (dist_next * (b - l.float())).view(-1),
            )

        # current distribution
        dist, _ = self.dqn_network(samples["state"], return_prob=True)

        # select the distribution for the action taken
        log_p = torch.log(dist[range(self.batch_size), samples["action"].squeeze()])
        # cross entropy loss
        return -(proj * log_p).sum(1)  # shape (batch_size, atom_size)

    def _target_hard_update(self):
        """Every target_update_freq steps, target_net <- copy(current_net)"""
        self.dqn_target.load_state_dict(self.dqn_network.state_dict())

    def train(self, num_episodes, show_progress=True):
        window_size = min(10, num_episodes // 10)
        rewards = []
        if self.agent_config["usePrioritized"]:
            beta_start = self.beta
            BETA_END = 1.0
            total_max_steps = num_episodes * NUMBER_STEPS
            total_max_steps = num_episodes * 200

        episode_bar = tqdm(total=num_episodes, desc="Episodes", leave=False)

        if self.agent_config["useDistributive"]:
            # Use fixed state for consistent distribution comparison
            fixed_state, _ = self.env.reset()

        for episode in range(num_episodes):
            state, _ = self.env.reset()

            done = False
            ep_reward = 0
            steps_n = 0

            while not done:
                if self.testing:
                    action = self.select_action(state)
                    next_state, reward, terminated, trucated, _ = self.env.step(action)
                    done = terminated or trucated
                else:
                    action, next_state, reward, done = self.step(state)

                    # only update if batch has enough samples
                    if len(self.memory) >= self.batch_size:
                        loss = self.update_model()

                    if self.total_steps % self.target_update_freq == 0:
                        self._target_hard_update()

                state = next_state
                ep_reward += float(reward)
                steps_n += 1

                if self.agent_config["usePrioritized"] and not self.testing:
                    # PER specific: annealed_beta = beta * fraction where fraction is steps left in total
                    fraction = min(self.total_steps / total_max_steps, BETA_END)  # type: ignore
                    self.beta = beta_start + fraction * (BETA_END - beta_start)  # type: ignore

            if (
                len(rewards) == (num_episodes - NUMBER_TEST_EPISODES)
                and not self.testing
            ):
                print("flipped to testing")
                self.testing = True

            if self.agent_config["useDistributive"] and not self.testing:
                # Track distributions periodically using the fixed state
                if episode % self.tracking_interval == 0:
                    self.track_distribution(fixed_state, episode)  # type: ignore

            rewards.append(ep_reward)
            # rewards2d.append(ep_rewards)
            # Calculate moving average

            recent_rewards = (
                rewards[-window_size:] if len(rewards) >= window_size else rewards
            )
            avg_reward = np.mean(recent_rewards)

            # Same progress bar update as before
            postfix_dict = {
                "reward": f"{ep_reward:.1f}",
                "avg": f"{avg_reward:.1f}",
                "steps": steps_n,
            }

            if not self.agent_config.get("useNoisy", False):
                postfix_dict["ε"] = f"{self.epsilon:.3f}"

            # buffer_size = len(self.memory)
            # postfix_dict['buffer'] = f"{buffer_size}/{MEMORY_SIZE}"

            episode_bar.update(1)
            episode_bar.set_postfix(postfix_dict)
            # episode_bar.update(1)
            # episode_bar.set_postfix(reward=f"{ep_reward:.1f}", steps=steps_n,
            #                         epsilon=f"{self.epsilon:.2f}", rews_avg=f"{np.mean(rewards):.2f}")

        if show_progress and episode_bar is not None:
            episode_bar.close()
        self.env.close()

        # if self.agent_config["useDistributive"]: # TODO may want to enable later
        #     self.plot_distribution_evolution()

        return rewards

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
        plt.imshow(
            dist_array,
            aspect="auto",
            extent=[
                support[0],
                support[-1],
                self.distribution_episodes[-1],
                self.distribution_episodes[0],
            ],  # type: ignore
            cmap="viridis",
            interpolation="nearest",
        )

        plt.colorbar(label="Probability")
        plt.xlabel("Return Value")
        plt.ylabel("Episode")
        plt.title("Evolution of Value Distribution During Training")

        # Also create a 3D visualization showing select distributions
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot a subset of distributions (to avoid overcrowding)
        max_to_show = min(10, num_distributions)
        indices = np.linspace(0, num_distributions - 1, max_to_show, dtype=int)

        # Create x, y coordinates for the 3D plot
        x = np.arange(len(support))
        for i in indices:
            y = self.distribution_episodes[i]
            z = self.distribution_history[i]
            ax.bar(x, z, zs=y, zdir="y", alpha=0.8, width=0.8)

        ax.set_xlabel("Atom Index")
        ax.set_ylabel("Episode")
        ax.set_zlabel("Probability")  # type: ignore
        ax.set_title("Value Distributions During Training")

        # Set x-ticks to show atom values
        step = max(1, len(support) // 10)  # Show at most 10 ticks
        ax.set_xticks(np.arange(0, len(support), step))
        ax.set_xticklabels([f"{v:.1f}" for v in support[::step]])

        # Save figures
        os.makedirs("distribution_plots", exist_ok=True)
        plt.savefig(f"distribution_plots/distribution_evolution.png")
        plt.close()
        # plt.show()
