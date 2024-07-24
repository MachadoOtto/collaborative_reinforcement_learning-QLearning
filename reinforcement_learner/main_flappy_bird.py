import gymnasium as gym
import flappy_bird_gymnasium
from discrete_agent.Agent import Agent
from utils import plotLearning
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    env = gym.make("FlappyBird-v0", use_lidar=False)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=2, eps_end=0.01,
                  input_dims=[12], lr=0.001)
    scores, eps_history = [], []
    n_games = 1500
    
    for i in tqdm(range(n_games)):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        # print('episode ', i, 'score %.2f' % score,
        #         'average score %.2f' % avg_score,
        #         'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'flappy.png'
    plotLearning(x, scores, eps_history, filename)

    # Now play the game with the trained agent
    env = gym.make('FlappyBird-v0', render_mode='human', use_lidar=False)
    for i in range(5):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)
            score += reward
            observation = observation_
        print('score ', score)
    env.close()
