import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import gym_maze
from GreedyPolicies import *
from BoltzmannPolicies import *
from QLearningAgent import QLearningAgent
from SarsaAgent import SarsaAgent
from PolicyGradientAgent import PolicyGradientAgent

env = gym.make('Maze-v0')
greed = GreedyPolicy()
e_greedy = EGreedyPolicy(decay=True, epsilon=0.3)
boltzmann = BoltzmannPolicy()
boltzmann_e_greedy = BoltzmannEGreedyPolicy()


def train_ql_agent():
    q_table_file_name = "q_learning_table.pkl"
    qLearningAgent = QLearningAgent(env=env, policy=e_greedy)
    qLearningAgent.load_stored_q_table(q_table_file_name)
    qLearningAgent.start_training(
        num_of_episodes=5,
        time_between_step=0.1,
        time_between_episode=1,
        save_q_table=True,
        q_table_file_name=q_table_file_name
    )


def train_sarsa_agent():
    sarsa_file_name = "sarsa_table.pkl"
    sarsaAgent = SarsaAgent(env=env, policy=e_greedy)
    sarsaAgent.load_stored_q_table(sarsa_file_name)
    sarsaAgent.start_training(
        num_of_episodes=5,
        time_between_step=0.1,
        time_between_episode=1,
        save_q_table=True,
        q_table_file_name=sarsa_file_name
    )


def train_policy_gradient_agent():
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    - alpha : learning rate for the policy parameters (theta)
    - gamma : discount factor
    No external policy object is needed; the agent owns its softmax policy.
    """
    pg_file_name = "policy_gradient_theta.pkl"
    pgAgent = PolicyGradientAgent(env=env, alpha=0.01, gamma=0.99)
    pgAgent.load_stored_theta(pg_file_name)
    pgAgent.start_training(
        num_of_episodes=5,
        time_between_step=0.1,
        time_between_episode=1,
        save_theta=True,
        theta_file_name=pg_file_name
    )


def main():
    print("=" * 40)
    print("  Training Q-Learning Agent")
    print("=" * 40)
    train_ql_agent()

    print("=" * 40)
    print("  Training SARSA Agent")
    print("=" * 40)
    train_sarsa_agent()

    print("=" * 40)
    print("  Training Policy Gradient (REINFORCE) Agent")
    print("=" * 40)
    train_policy_gradient_agent()


main()
