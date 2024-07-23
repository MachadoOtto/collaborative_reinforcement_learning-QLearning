import gymnasium as gym
import numpy as np
import sys

def train(episodes):
    env = gym.make("MountainCar-v0")

    # Divide position and velocity into segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=20)

    q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))

    learning_rate_a = 0.9
    discount_rate_g = 0.9

    epsilon = 1
    epsilon_decay = 2 / episodes
    rng = np.random.default_rng()

    for episode in range(episodes):
        state = env.reset()[0]

        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False
        truncation = False

        rewards = 0

        while not terminated and rewards > -1000:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p][state_v])

            new_state, reward, terminated, truncation, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            q[state_p][state_v][action] = q[state_p][state_v][action] + learning_rate_a * (reward + discount_rate_g * np.max(q[new_state_p][new_state_v]) - q[state_p][state_v][action])

            state_p = new_state_p
            state_v = new_state_v
            state = new_state

            rewards += reward

        epsilon = max(epsilon - epsilon_decay, 0)

        # Calcular y mostrar el progreso
        progress = (episode + 1) / episodes * 100
        sys.stdout.write(f"\rProgreso: [{int(progress)}%]")
        sys.stdout.flush()

    env.close()
    print("\nEntrenamiento completado.")
    return q, pos_space, vel_space

def play(q, pos_space, vel_space):
    env = gym.make("MountainCar-v0", render_mode="human")
    state = env.reset()[0]

    state_p = np.digitize(state[0], pos_space)
    state_v = np.digitize(state[1], vel_space)

    terminated = False
    truncation = False

    rewards = 0

    while not terminated and rewards > -1000:
        action = np.argmax(q[state_p][state_v])
        new_state, reward, terminated, truncation, _ = env.step(action)
        new_state_p = np.digitize(new_state[0], pos_space)
        new_state_v = np.digitize(new_state[1], vel_space)

        state_p = new_state_p
        state_v = new_state_v
        state = new_state

        rewards += reward

    env.close()

if __name__ == "__main__":
    episodes = 500  # Número de episodios para entrenamiento
    print("Training for", episodes, "episodes")
    q, pos_space, vel_space = train(episodes)
    play(q, pos_space, vel_space)