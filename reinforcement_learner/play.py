import importlib
import logging
import os
from argparse import ArgumentParser

import config
import gymnasium as gym
import numpy as np
import torch
from discrete_agent.AgentV2 import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def watch_n(env, agent, n: int = 5):
    agent.eval()

    for _ in range(n):
        done = False
        score = 0
        state, _ = env.reset()
        state = torch.tensor(
            state, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)
        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=config.DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=config.DEVICE
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
        state = torch.tensor(
            state, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)
        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=config.DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=config.DEVICE
                ).unsqueeze(0)

            state = next_state
            score += reward.item()

        scores.append(score)

    print(
        f"Episodes: {n}, Average score: {np.mean(scores)}, Median score: {np.median(scores)}, Max score: {np.max(scores)}, Min score: {np.min(scores)}"
    )


def main(env_name: str, model: str, **kwargs) -> None:
    if module := config.CONFIGS[env_name]["env"]["import"]:
        importlib.import_module(module)
    env = gym.make(
        config.CONFIGS[env_name]["env"]["name"],
        **config.CONFIGS[env_name]["env"]["kwargs"],
    )
    ag = Agent(
        **config.TUTORIAL,
        env=env,
    )
    # EL MERGE NO ESTA FUNCIONANDO BIEN
    ag.load_model(
        f"{config.OUT_DIR}/models/{model}", from_checkpoint=model.endswith(".tar")
    )

    ### TODO: ifs que comando selecciono eval, o bof, o record

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
    arg_parser = ArgumentParser()
    # required
    arg_parser.add_argument(
        "--env_name", type=str, required=True, help="Environment to use"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to load. Ex. 'LunarLander-v2-5000-base1.tar'",
    )
    # optional

    args = arg_parser.parse_args()
    logging.debug("Script called with args: %s", args)
    main(**vars(args))
