import os
from statistics import median

import config
import gymnasium as gym
import numpy as np
import torch
from discrete_agent.AgentV2 import Agent

OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"


def watch_n(env, agent, n: int = 5):
    agent.eval()

    for _ in range(n):
        done = False
        score = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device="cuda:0")
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device="cuda:0"
                ).unsqueeze(0)

            state = next_state
            score += reward.item()
        print("score ", score)


def record_bof_n(env_name, agent, n: int = 100):
    tmp_env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        tmp_env,
        f"{OUT_DIR}/videos",
        name_prefix="LunarLander-v2",
        episode_trigger=lambda x: True,
    )
    agent.eval()

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


def evaluate_n(env, agent, n: int = 500):
    agent.eval()
    scores = []

    for _ in range(n):
        done = False
        score = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device="cuda:0")
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device="cuda:0"
                ).unsqueeze(0)

            state = next_state
            score += reward.item()

        scores.append(score)

    avg_score = np.mean(scores)
    median_score = median(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    print(
        f"Average score: {avg_score}, Median score: {median_score}, Max score: {max_score}, Min score: {min_score}"
    )


def main():
    env_name = "FlappyBird-v0"
    if env_name == "FlappyBird-v0":
        import flappy_bird_gymnasium  # noqa: F401

    env = gym.make(
        config.CONFIGS[env_name]["env"]["name"],
        **config.CONFIGS[env_name]["env"]["kwargs"],
    )
    ag = Agent(
        **config.TUTORIAL,
        env=env,
    )
    # ag.load_model(
    #     f"{OUT_DIR}/models/{env_name}-base1-base2_merged_model.pt"
    # )  # EL MERGE NO FUNCIONA BIEN
    ag.load_model(f"{OUT_DIR}/models/{env_name}-5000-base3.tar", from_checkpoint=True)

    # record_bof_n(env_name="LunarLander-v2", agent=ag)
    evaluate_n(env=env, agent=ag)

    env.close()
    env2 = gym.make(
        config.CONFIGS[env_name]["env"]["name"],
        **config.CONFIGS[env_name]["env"]["kwargs"],
        render_mode="human",
    )
    watch_n(env=env2, agent=ag)
    env2.close()


if __name__ == "__main__":
    main()
