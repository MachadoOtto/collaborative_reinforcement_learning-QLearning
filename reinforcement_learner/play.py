import os
from statistics import median

import gymnasium as gym
import numpy as np
from discrete_agent.Agent import Agent

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"


def watch_n(env_name, agent, n: int = 5):
    env = gym.make(env_name, render_mode="human")
    agent.Q_eval.eval()

    for _ in range(n):
        done = False
        score = 0
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            observation = observation_

        print("score ", score)

    env.close()


def record_bof_n(env_name, agent, n: int = 100):
    tmp_env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        tmp_env,
        f"{OUT_DIR}/videos",
        name_prefix="LunarLander-v2",
        episode_trigger=lambda x: True,
    )
    agent.Q_eval.eval()

    for _ in range(n):
        done = False
        score = 0
        observation, _ = env.reset()
        env.start_video_recorder()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            observation = observation_
            score += reward
            env.render()

        print("score ", score)
        if score <= 0:
            os.remove(env.video_recorder.path)
            os.remove(env.video_recorder.metadata_path)

    env.close()


def evaluate_n(env_name, agent, n: int = 500):
    env = gym.make(env_name)
    agent.Q_eval.eval()

    scores = []

    for _ in range(n):
        done = False
        score = 0
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
            observation = observation_

        scores.append(score)

    avg_score = np.mean(scores)
    median_score = median(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    print(
        f"Average score: {avg_score}, Median score: {median_score}, Max score: {max_score}, Min score: {min_score}"
    )
    env.close()


def main():
    ag = Agent(
        gamma=0.99,
        epsilon=1.0,
        eps_dec=0.99941,
        eps_end=0.01,
        batch_size=64,
        n_actions=4,
        input_dims=[8],
        lr=0.0001,
    )
    # ag.load_model(f"{OUT_DIR}/models/merged_model.pt")
    ag.load_model(
        f"{OUT_DIR}/models/LunarLander-v2-10000-version4.tar", from_checkpoint=True
    )

    # record_bof_n(env_name="LunarLander-v2", agent=ag)
    evaluate_n(env_name="LunarLander-v2", agent=ag)
    watch_n(env_name="LunarLander-v2", agent=ag)


if __name__ == "__main__":
    main()
