import math
import random
from collections import deque, namedtuple

import config
import torch as T
import torch.nn as nn
import torch.optim as optim
from deep_q_network.DeepQNetworkV2 import DQN

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, env):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = DQN(self.n_observations, self.n_actions).to(config.DEVICE)
        self.target_net = DQN(self.n_observations, self.n_actions).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def eps_threshold(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )

    def choose_action(self, state):
        sample = random.random()
        self.steps_done += 1
        if sample > self.eps_threshold():
            with T.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return T.tensor(
                [[self.env.action_space.sample()]], device=config.DEVICE, dtype=T.long
            )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = T.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=config.DEVICE,
            dtype=T.bool,
        )
        non_final_next_states = T.cat([s for s in batch.next_state if s is not None])
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = T.zeros(self.batch_size, device=config.DEVICE)
        with T.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        T.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def save_checkpoint(self, episode: int, path: str, model_name: str) -> None:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        T.save(
            {
                "episode": episode,
                "steps_done": self.steps_done,
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "memory": self.memory,
            },
            f"{path}/{model_name}.tar",
        )

    def load_checkpoint(self, path: str) -> tuple[int, float]:
        checkpoint = T.load(path, map_location=T.device("cpu"))

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.memory = checkpoint["memory"]

        self.steps_done = checkpoint["steps_done"]

        return checkpoint["episode"], self.eps_threshold()

    def save_model(self, path: str, model_name: str) -> None:
        """Save for inference. Only the model."""
        T.save(self.policy_net.state_dict(), f"{path}/{model_name}.pt")

    def load_model(self, path: str, from_checkpoint: bool = False) -> None:
        """
        Este se usa cuando el master le pase un modelo.
        Ahi no tiene sentido cargar el optimizador porque cada slave genero un estado del optimizador distinto, tiene que emepezar con un optimizador nuevo.
        """
        model_state_dict = T.load(path, map_location=T.device("cpu"))
        if from_checkpoint:
            model_state_dict = model_state_dict["model_state_dict"]

        self.policy_net.load_state_dict(model_state_dict)
        self.target_net.load_state_dict(model_state_dict)
