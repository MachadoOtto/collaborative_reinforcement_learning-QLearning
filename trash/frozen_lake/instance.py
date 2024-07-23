## Imports ##
import gymnasium as gym
import numpy as np
import sys
from tqdm import tqdm
import imageio

## Constants ##
ENVIRONMENT = "FrozenLake-v1"
SAVE_PATH = "./outputs/frozen-lake/q.npy"
EPISODES = 500

### Policies ###
GREEDY_POLICY = "exploitation" # "exploration" or "exploitation"

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.9
EPSILON = 1


# env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")
# desc=["SFHF", "FFFH", "FFHH", "HFFG"]
# env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False, render_mode="rgb_array")
# env = gym.make("Taxi-v3", render_mode="rgb_array")
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")

# Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros
def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

Qtable_frozenlake = initialize_q_table(state_space, action_space)

def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = np.random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state
    return Qtable

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print("Mean reward: ", mean_reward, "Std reward: ", std_reward)

def record_video(env, Qtable, out_directory, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=np.random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not terminated or truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, terminated, truncated, info = env.step(
            action
        )  # We directly put next_state = state for recording logic
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)

# Step 6: Record a video
video_path = "replay5.mp4"
record_video(env, Qtable_frozenlake, video_path, fps=1)

class FrozenLake:
    def __init__(self):
        self.env = gym.make(ENVIRONMENT, is_slippery=True)
        self.epsilon_decay = 2 / EPISODES
        self.rng = np.random.default_rng()
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def train(self):
        epsilon = EPSILON

        for episode in range(EPISODES):
            state = self.env.reset()[0]
            terminated = False
            rewards = 0

            while not terminated:
                if self.rng.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q[state])

                new_state, reward, terminated, _, _ = self.env.step(action)

                self.q[state][action] = self.q[state][action] + LEARNING_RATE * (
                    reward + DISCOUNT_RATE * np.max(self.q[new_state]) - self.q[state][action]
                )

                state = new_state
                rewards += reward

            epsilon = max(epsilon - self.epsilon_decay, 0)

            # Calcular y mostrar el progreso
            progress = (episode + 1) / EPISODES * 100
            sys.stdout.write(f"\rProgreso: [{int(progress)}%]")
            sys.stdout.flush()

        self.env.close()
        print("\nEntrenamiento completado.")
        return self.q

    def save(self):
        np.save(SAVE_PATH, self.q)
        print("Modelo guardado en", SAVE_PATH)

    def load(self):
        self.q = np.load(SAVE_PATH)
        print("Modelo cargado de", SAVE_PATH)

    def play(self):
        self.env = gym.make(ENVIRONMENT, render_mode="human")
        state = self.env.reset()[0]
        terminated = False

        while not terminated:
            action = np.argmax(self.q[state])
            state, _, terminated, _, _ = self.env.step(action)

        self.env.close()
        print("Juego completado.")

# if __name__ == "__main__":
#     frozen_lake = FrozenLake()
#     frozen_lake.train()
#     frozen_lake.save()
#     frozen_lake.load()
#     frozen_lake.play()
