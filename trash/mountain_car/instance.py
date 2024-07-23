# Instance MountainCar

## Imports
import gymnasium as gym
import numpy as np
import sys

## Constants
ENVIRONMENT = "MountainCar-v0"
SAVE_PATH = "./outputs/mountain-car/q.npy"
EPISODES = 500

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.9
EPSILON = 1

## Class
class MountainCar:
    def __init__(self):
        self.env = gym.make(ENVIRONMENT)
        self.epsilon_decay = 2 / EPISODES
        self.rng = np.random.default_rng()

        # Divide position and velocity into segments
        self.pos_space = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], num=20)
        self.vel_space = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], num=20)
        self.q = np.zeros((len(self.pos_space), len(self.vel_space), self.env.action_space.n))

    def train(self):
        epsilon = EPSILON

        for episode in range(EPISODES):
            state = self.env.reset()[0]

            state_p = np.digitize(state[0], self.pos_space)
            state_v = np.digitize(state[1], self.vel_space)

            terminated = False
            rewards = 0

            while not terminated and rewards > -1000:
                if self.rng.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q[state_p][state_v])

                new_state, reward, terminated, _, _ = self.env.step(action)
                new_state_p = np.digitize(new_state[0], self.pos_space)
                new_state_v = np.digitize(new_state[1], self.vel_space)

                self.q[state_p][state_v][action] = self.q[state_p][state_v][action] + LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(self.q[new_state_p][new_state_v]) - self.q[state_p][state_v][action])

                state_p = new_state_p
                state_v = new_state_v
                state = new_state

                rewards += reward

            epsilon = max(epsilon - self.epsilon_decay, 0)

            # Calcular y mostrar el progreso
            progress = (episode + 1) / EPISODES * 100
            sys.stdout.write(f"\rProgreso: [{int(progress)}%]")
            sys.stdout.flush()

        self.env.close()
        print("\nEntrenamiento completado.")
        return self.q, self.pos_space, self.vel_space
    
    def save(self):
        np.save(SAVE_PATH, self.q)
        print("Modelo guardado en", SAVE_PATH)

    def load(self):
        self.q = np.load(SAVE_PATH)
        print("Modelo cargado de", SAVE_PATH)

    def play(self):
        self.env = gym.make(ENVIRONMENT, render_mode="human")
        state = self.env.reset()[0]
        state_p = np.digitize(state[0], self.pos_space)
        state_v = np.digitize(state[1], self.vel_space)

        terminated = False

        while not terminated:
            action = np.argmax(self.q[state_p][state_v])
            new_state, _, terminated, _, _ = self.env.step(action)
            new_state_p = np.digitize(new_state[0], self.pos_space)
            new_state_v = np.digitize(new_state[1], self.vel_space)

            state_p = new_state_p
            state_v = new_state_v

        self.env.close()
        print("Juego completado.")

if __name__ == "__main__":
    mountain_car = MountainCar()
    # mountain_car.train()
    # mountain_car.save()
    mountain_car.load()
    mountain_car.play()

