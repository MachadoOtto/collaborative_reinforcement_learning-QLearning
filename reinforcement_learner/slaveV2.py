import importlib
import logging
import os
from argparse import ArgumentParser

import config
import gymnasium as gym
import torch
from discrete_agent.AgentV2 import Agent
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def train(
    agent: Agent,
    env: gym.Env,
    n_games: int,
) -> tuple[gym.Env, Agent, dict]:
    scores, eps_history = [], []

    for _ in tqdm(range(n_games)):
        score = 0
        done = False
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

            agent.memory.push(state, action, next_state, reward)
            agent.learn()
            state = next_state

            score += reward.item()
        scores.append(score)
        eps_history.append(agent.eps_threshold)

    statistics = {"scores": scores, "epsilons": eps_history}
    return agent, statistics


def main(env_name: str, n_games: int, **kwargs):
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

    # maybe use satistics wrapper
    ag, statistics = train(agent=ag, env=env, n_games=n_games)
    env.close()

    total_episodes = episode + n_games
    model_name = f"{env_name}-{total_episodes}"  # TODO: pensar un mejor sistema de versionado de nombres, algun hash o algo para saber tmbn cual es el modelo base

    out_path = f"{config.OUT_DIR}/models"
    if kwargs.get("save_checkpoint"):
        ag.save_checkpoint(
            episode=total_episodes,
            path=out_path,
            model_name=f"{model_name}{kwargs.get("save_suffix", "")}",
        )
        logging.info("Model %s checkpointed at %s", model_name, out_path)
    if kwargs.get("save_model"):
        ag.save_model(
            out_path, model_name=f"{model_name}{kwargs.get("save_suffix", "")}"
        )
        logging.info("Model %s saved at %s", model_name, out_path)

    # save statistics as csv (appends if exists)
    csv_file = f"{config.OUT_DIR}/stats/{env_name}{kwargs.get("save_suffix", "")}.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("episode,score,epsilon\n")
        for i, (score, epsilon) in enumerate(
            zip(statistics["scores"], statistics["epsilons"])
        ):
            f.write(f"{episode + i},{score},{epsilon}\n")
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
        "--save_suffix",
        type=str,
        default=None,
        help="Suffix to add to the saved model/checkpoint name",
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
