import importlib
import logging
import os
from argparse import ArgumentParser

import config
import gymnasium as gym
import torch
from discrete_agent.Agent import Agent
from tqdm import tqdm
from utils import StatPoint, save_stats_to_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def train(
    agent: Agent,
    env: gym.Env,
    n_games: int,
) -> tuple[gym.Env, Agent, dict]:
    agent.train()
    stats = []

    for _ in tqdm(range(n_games)):
        done = False
        state, _ = env.reset()
        state = torch.tensor(
            state, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)
        while not done:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=config.DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=config.DEVICE
                ).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)
            agent.learn()
            state = next_state

        data = info["episode"]
        stats.append(StatPoint(*data["r"], *data["l"], *data["t"], agent.eps_threshold))

    return agent, stats


def main(env_name: str, n_games: int, **kwargs):
    if module := config.CONFIGS[env_name]["env"].get("import"):
        importlib.import_module(module)

    env = gym.make(
        config.CONFIGS[env_name]["env"]["name"],
        **config.CONFIGS[env_name]["env"]["kwargs"],
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    ag = Agent(
        **config.TUTORIAL,
        env=env,
    )

    # initialize variables or set from pretrained model
    episode, epsilon = 0, 1.0
    if kwargs.get("base_checkpoint"):
        base_checkpoint = kwargs.get("base_checkpoint")
        episode, epsilon = ag.load_checkpoint(path=base_checkpoint)
        logging.info(
            "Checkpoint loaded from %s. With params: episode %s, epsilon %s",
            base_checkpoint,
            episode,
            epsilon,
        )
    elif kwargs.get("base_model"):
        base_model = kwargs.get("base_model")
        ag.load_model(base_model)
        logging.info("Model loaded from %s", base_model)

    ag, statistics = train(agent=ag, env=env, n_games=n_games)
    env.close()

    total_episodes = episode + n_games
    
    file_name = kwargs.get("file_name") or env_name
    out_path = f"{config.OUT_DIR}/models/{file_name}"
    if kwargs.get("save_checkpoint"):
        ag.save_checkpoint(episode=total_episodes, path=out_path)
        logging.info("Model checkpointed at %s", out_path)
    if kwargs.get("save_model"):
        ag.save_model(out_path)
        logging.info("Model saved at %s", out_path)

    csv_file = f"{config.OUT_DIR}/stats/{file_name}.csv"
    save_stats_to_csv(statistics, csv_file, write_header=not os.path.exists(csv_file))
    logging.info("Statistics saved at %s", csv_file)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    # required
    arg_parser.add_argument(
        "--env_name", type=str, required=True, help="Environment to use"
    )
    arg_parser.add_argument(
        "--n_games", type=int, required=True, help="Number of games to play"
    )
    # optional
    arg_parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        help="Wheter to save the model weights or not",
    )
    arg_parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=False,
        help="Wheter to save a full checkpoint or not",
    )
    arg_parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Name of the file to save the model or checkpoint. Default is the current environment name",
    )

    load_group = arg_parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--base_model", type=str, default=None, help="Path to a model to load"
    )
    load_group.add_argument(
        "--base_checkpoint", type=str, default=None, help="Path to a checkpoint to load"
    )

    args = arg_parser.parse_args()
    logging.debug("Script called with args: %s", args)
    main(**vars(args))
