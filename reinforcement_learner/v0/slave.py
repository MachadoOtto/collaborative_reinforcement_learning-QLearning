import logging
import os
from argparse import ArgumentParser

import gymnasium as gym
import numpy as np
from reinforcement_learner.v0.discrete_agent.Agent import Agent
from tqdm import tqdm

from ..config import CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# TODO: ver como hacer para que no dependa de donde se ejecute
OUT_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/outputs"


def train(
    agent: Agent,
    env: gym.Env,
    n_games: int,
) -> tuple[gym.Env, Agent, dict]:
    scores, eps_history = [], []

    for _ in tqdm(range(n_games)):
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

    statistics = {"scores": scores, "epsilons": eps_history}
    return agent, statistics


def main(env_name: str, n_games: int, **kwargs):
    env = gym.make(
        CONFIGS[env_name]["env"]["name"], **CONFIGS[env_name]["env"]["kwargs"]
    )
    ag = Agent(
        **CONFIGS[env_name]["agent"]["hyperparams"],
        **CONFIGS.DQN,
        batch_size=CONFIGS.BATCH_SIZE,
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
    )

    # initialize variables or set from pretrained model
    episode, loss, epsilon = 0, -np.inf, 1.0
    if kwargs.get("base_checkpoint"):
        base_checkpoint = kwargs.get("base_checkpoint")
        episode, loss, epsilon = ag.load_checkpoint(path=base_checkpoint)
        logging.info(
            "Checkpoint loaded from %s. With params: epoch %s, loss %s, epsilon %s",
            base_checkpoint,
            episode,
            loss,
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

    out_path = kwargs.get("out_path") or f"{config.OUT_DIR}/models"
    if kwargs.get("save_checkpoint"):
        ag.save_checkpoint(episode=total_episodes, path=out_path, model_name=model_name)
        logging.info("Model %s checkpointed at %s", model_name, out_path)
    if kwargs.get("save_model"):
        ag.save_model(out_path, model_name=model_name)
        logging.info("Model %s saved at %s", model_name, out_path)

    csv_file = f"{config.OUT_DIR}/stats/{env_name}.csv"
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
        "--out_path",
        type=str,
        default=None,
        help="Path to save the model/checkpoint. Default: '{OUT_DIR}/models/'",
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
