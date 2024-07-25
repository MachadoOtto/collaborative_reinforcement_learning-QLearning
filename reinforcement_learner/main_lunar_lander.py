import logging
import os
import gymnasium as gym
import numpy as np
from discrete_agent.Agent import Agent
from tqdm import tqdm
from utils import plotLearning

logging.basicConfig(level=logging.INFO)

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"


def format_score(score: float):
    return f"{score:.2f}".replace(".", "_")


def train(
    agent: Agent,
    env: gym.Env,
    n_games: int,
    checkpoint: bool = True,
    epoch: int = 0,
    loss: float = -np.inf,
    plot: bool = True,
):
    env_name = env.unwrapped.spec.id
    scores, eps_history = [], []
    best_score = loss

    for episode in tqdm(range(n_games)):
        score = 0
        done = False
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            done = terminated or truncated
        scores.append(score)
        eps_history.append(agent.epsilon)

        # checkpoint the model if the score is better than the last checkpoint
        if checkpoint and episode > n_games // 2 and score > best_score:
            best_score = score
            agent.save_model(
                epoch=episode,
                loss=best_score,
                path=f"{OUT_DIR}/models/{env_name}-{episode+epoch}-{format_score(best_score)}",
            )
            logging.info(
                f"Episode {episode+epoch} with Best Score {best_score} checkpointed"
            )

        # avg_score = np.mean(scores[-100:])

        # print(
        #     "episode ",
        #     i,
        #     "score %.2f" % score,
        #     "average score %.2f" % avg_score,
        #     "epsilon %.2f" % agent.epsilon,
        # )
    env.close()

    if plot:
        x = [i + 1 for i in range(n_games)]
        plotLearning(
            x, scores, eps_history, f"{OUT_DIR}/plots/{env_name}-{n_games}.png"
        )


def play(env, agent):
    agent.Q_eval.eval()
    for _ in range(5):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, _, _ = env.step(action)
            score += reward
            observation = observation_
        print("score ", score)
    env.close()


if __name__ == "__main__":
    ENV_NAME = "LunarLander-v2"
    # maybe use satistics wrapper

    ag = Agent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=4,
        eps_end=0.01,
        input_dims=[8],
        lr=0.001,
    )
    epoch, loss = ag.load_model(f"{OUT_DIR}/models/{ENV_NAME}-1208-307_02.tar")

    train(agent=ag, env=gym.make(ENV_NAME), n_games=500, epoch=epoch + 1, loss=loss)

    # play(gym.make(env_name, render_mode="human"), ag)
