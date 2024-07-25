import numpy as np
import torch as T
from deep_q_network.DeepQNetwork import DeepQNetwork


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_dims,
        batch_size,
        n_actions,
        max_mem_size=100000,
        eps_end=0.05,
        eps_dec=5e-4,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(
            lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256
        )
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(observation), dtype=T.float32).to(
                self.Q_eval.device
            )
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch], dtype=T.float32).to(
            self.Q_eval.device
        )
        new_state_batch = T.tensor(self.new_state_memory[batch], dtype=T.float32).to(
            self.Q_eval.device
        )
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch], dtype=T.float32).to(
            self.Q_eval.device
        )
        terminal_batch = T.tensor(self.terminal_memory[batch], dtype=T.bool).to(
            self.Q_eval.device
        )

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)

    def save_checkpoint(
        self, episode: int, epsilon: float, path: str, model_name: str
    ) -> None:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        T.save(
            {
                "episode": episode,
                "epsilon": epsilon,
                "loss": self.Q_eval.loss,
                "model_state_dict": self.Q_eval.state_dict(),
                "optimizer_state_dict": self.Q_eval.optimizer.state_dict(),
            },
            f"{path}/{model_name}.tar",
        )

    def load_checkpoint(self, path: str) -> tuple[int, float, float]:
        checkpoint = T.load(path, map_location=T.device("cpu"))

        self.Q_eval.load_state_dict(checkpoint["model_state_dict"])
        self.Q_eval.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint["epsilon"]
        return checkpoint["episode"], checkpoint["loss"], checkpoint["epsilon"]

    def save_model(self, path: str, model_name: str) -> None:
        T.save(self.Q_eval.state_dict(), f"{path}/{model_name}.pt")

    def load_model(self, path: str, from_checkpoint: bool = False) -> None:
        """
        Este se usa cuando el master le pase un modelo.
        Ahi no tiene sentido cargar el optimizador porque cada slave genero un estado del optimizador distinto, tiene que emepezar con un optimizador nuevo.
        """
        model_state_dict = T.load(path, map_location=T.device("cpu"))
        if from_checkpoint:
            model_state_dict = model_state_dict["model_state_dict"]
        self.Q_eval.load_state_dict(model_state_dict)
