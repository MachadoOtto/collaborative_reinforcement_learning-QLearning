import importlib
import logging
import os
from argparse import ArgumentParser

import config
import gymnasium as gym
import numpy as np
import torch
from discrete_agent.Agent import Agent

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
        env.get_wrapper_attr("start_video_recorder")

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
        f"{config.OUT_DIR}/videos/{env_kwargs["env"]["name"]}",
        name_prefix=env_kwargs["env"]["name"],
        episode_trigger=lambda x: True,
        disable_logger=True,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent.eval()

    datas = []
    best_path, best_metadata, best_score = None, None, -np.inf
    for episode in range(n):
        data = run_episode(env, agent, record_video=True)
        datas.append(data)
        score = data["r"][0]
        print(f"Episode {episode + 1}/{n}, Score: {score}")

        video_recorder = env.get_wrapper_attr("video_recorder")
        current_video_path = video_recorder.path
        current_metadata_path = video_recorder.metadata_path

        if score > best_score:
            print(f"New best score: {score}")
            if best_path and os.path.exists(best_path):
                os.remove(best_path)
            if best_metadata and os.path.exists(best_metadata):
                os.remove(best_metadata)

            best_score = score
            best_path = current_video_path
            best_metadata = current_metadata_path
        else:
            if os.path.exists(current_video_path):
                os.remove(current_video_path)
            if os.path.exists(current_metadata_path):
                os.remove(current_metadata_path)

    print_statistics(datas, n)
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

    ag.load_model(model, from_checkpoint=model.endswith(".tar"))

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
        help="Record best episode over <n> episodes. Default: 100",
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
