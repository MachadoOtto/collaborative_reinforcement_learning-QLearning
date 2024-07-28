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


def create_env(env_kwargs, render_mode=None):
    kwargs = env_kwargs["env"]["kwargs"].copy()
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(env_kwargs["env"]["name"], **kwargs)


def run_episode(env, agent, record_video=False):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=config.DEVICE).unsqueeze(0)
    done = False

    if record_video:
        env.start_video_recorder()

    while not done:
        action = agent.choose_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=config.DEVICE)
        done = terminated or truncated

        next_state = (
            None
            if terminated
            else torch.tensor(
                observation, dtype=torch.float32, device=config.DEVICE
            ).unsqueeze(0)
        )

        state = next_state

        if record_video:
            env.render()

    return info["episode"]


def watch(env_kwargs, agent, n=5):
    env = create_env(env_kwargs, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent.eval()

    for _ in range(n):
        data = run_episode(env, agent)
        print("score ", *data["r"])

    env.close()


def record(env_kwargs, agent, n=100):
    tmp_env = create_env(env_kwargs, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        tmp_env,
        f"{config.OUT_DIR}/videos",
        name_prefix=env_kwargs["env"]["name"],
        episode_trigger=gym.wrappers.capped_cubic_video_schedule,
    )
    agent.eval()

    for _ in range(n):
        data = run_episode(env, agent, record_video=True)
        score = data["r"][0]
        print("score ", score)
        if score <= 0:
            os.remove(env.video_recorder.path)
            os.remove(env.video_recorder.metadata_path)

    env.close()


def evaluate(env_kwargs, agent, n=500):
    tmp_env = create_env(env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(tmp_env)
    agent.eval()

    data = [run_episode(env, agent) for _ in range(n)]

    print_statistics(data, n)
    env.close()


def print_statistics(data, n):
    print("Total Episodes: ", n)
    for name, values in zip(
        [
            "cumulative reward",
            "episode length",
            "elapsed time since beginning of episode",
        ],
        zip(*[(d["r"], d["l"], d["t"]) for d in data]),
    ):
        print(name)
        print("mean=", np.mean(values))
        print("std=", np.std(values))
        print("min=", np.min(values))
        print("max=", np.max(values))


def main(env_name: str, model: str, **kwargs) -> None:
    if module := config.CONFIGS[env_name]["env"].get("import"):
        importlib.import_module(module)

    env = create_env(config.CONFIGS[env_name])
    ag = Agent(**config.TUTORIAL, env=env)
    env.close()

    ag.load_model(
        f"{config.OUT_DIR}/models/{model}", from_checkpoint=model.endswith(".tar")
    )

    if kwargs.get("record") is not None:
        record(env_kwargs=config.CONFIGS[env_name], agent=ag, n=kwargs["record"])

    if kwargs.get("evaluate") is not None:
        evaluate(env_kwargs=config.CONFIGS[env_name], agent=ag, n=kwargs["evaluate"])

    if kwargs.get("watch") is not None:
        watch(env_kwargs=config.CONFIGS[env_name], agent=ag, n=kwargs["watch"])


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--env_name", type=str, required=True, help="Environment to use"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to load. Ex. 'LunarLander-v2-5000-base1.tar'",
    )

    arg_parser.add_argument(
        "--record",
        type=int,
        help="Records some episodes.",
    )

    arg_parser.add_argument(
        "--evaluate",
        type=int,
        help="Evaluate the model over <n> episodes. Default: 500",
    )

    arg_parser.add_argument(
        "--watch", type=int, help="Watch the model play <n> games. Default: 5"
    )

    args = arg_parser.parse_args()
    logging.debug("Script called with args: %s", args)
    main(**vars(args))
