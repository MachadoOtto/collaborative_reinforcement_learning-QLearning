import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_learning_curve(scores, window_size=100):
    """
    Plot the learning curve with confidence intervals.

    :param scores: List of episode scores
    :param window_size: Size of the rolling window for averaging
    """
    x = np.arange(len(scores))
    means = np.convolve(scores, np.ones(window_size), "valid") / window_size
    std = np.array(
        [
            np.std(scores[i : i + window_size])
            for i in range(len(scores) - window_size + 1)
        ]
    )
    ci = stats.sem(scores) * std * 1.96  # 95% confidence interval

    plt.figure(figsize=(12, 6))
    plt.plot(x[window_size - 1 :], means, label="Rolling Mean")
    plt.fill_between(
        x[window_size - 1 :], means - ci, means + ci, alpha=0.2, label="95% CI"
    )
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Learning Curve with Confidence Intervals")
    plt.legend()
    plt.savefig("learning_curve.png")
    plt.close()


def plot_episode_length_distribution(episode_lengths):
    """
    Plot the distribution of episode lengths.

    :param episode_lengths: List of episode lengths
    """
    plt.figure(figsize=(10, 6))
    plt.hist(episode_lengths, bins=50, edgecolor="black")
    plt.xlabel("Episode Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Episode Lengths")
    plt.savefig("episode_length_distribution.png")
    plt.close()


def plot_reward_distribution(rewards):
    """
    Plot the distribution of rewards.

    :param rewards: List of episode rewards
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, edgecolor="black")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rewards")
    plt.savefig("reward_distribution.png")
    plt.close()


def plot_q_value_evolution(q_values, state_action_pairs):
    """
    Plot the evolution of Q-values for specific state-action pairs.

    :param q_values: List of dictionaries, each containing Q-values at a certain point in training
    :param state_action_pairs: List of (state, action) tuples to track
    """
    plt.figure(figsize=(12, 6))
    for state, action in state_action_pairs:
        values = [q[state][action] for q in q_values]
        plt.plot(values, label=f"State: {state}, Action: {action}")

    plt.xlabel("Training Step")
    plt.ylabel("Q-value")
    plt.title("Q-value Evolution for Key State-Action Pairs")
    plt.legend()
    plt.savefig("q_value_evolution.png")
    plt.close()


def plot_exploration_exploitation(explore_counts, exploit_counts):
    """
    Plot the proportion of exploratory vs exploitative actions over time.

    :param explore_counts: List of exploratory action counts per episode
    :param exploit_counts: List of exploitative action counts per episode
    """
    episodes = range(len(explore_counts))
    total_actions = np.array(explore_counts) + np.array(exploit_counts)
    explore_ratio = np.array(explore_counts) / total_actions
    exploit_ratio = np.array(exploit_counts) / total_actions

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        episodes, explore_ratio, exploit_ratio, labels=["Exploration", "Exploitation"]
    )
    plt.xlabel("Episode")
    plt.ylabel("Proportion of Actions")
    plt.title("Exploration vs Exploitation Over Time")
    plt.legend(loc="upper right")
    plt.savefig("exploration_exploitation.png")
    plt.close()


def plot_performance_comparison(scores_dict):
    """
    Plot performance comparison of different algorithms or hyperparameters.

    :param scores_dict: Dictionary of algorithm names to lists of scores
    """
    plt.figure(figsize=(12, 6))
    for name, scores in scores_dict.items():
        x = range(len(scores))
        plt.plot(x, scores, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Performance Comparison")
    plt.legend()
    plt.savefig("performance_comparison.png")
    plt.close()
