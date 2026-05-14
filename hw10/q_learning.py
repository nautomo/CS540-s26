import gymnasium as gym
import pickle
from collections import defaultdict
import random
import numpy as np

# Hyperparameters
EPISODES = 30000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MAX_STEPS = 200  # Maximum steps per episode


def default_Q_value():
    """Default Q-value for unseen state-action pairs"""
    return 0.0


def select_action(Q_table, state, n_actions, epsilon):
    """
    Select action using epsilon-greedy policy

    Args:
        Q_table: Dictionary mapping (state, action) to Q-values
        state: Current state
        n_actions: Number of available actions
        epsilon: Exploration probability

    Returns:
        Selected action (int)
    """
    ##########################################################
    # TODO: Implement epsilon-greedy action selection
    # With probability epsilon: return random action
    # Otherwise: return action with highest Q-value
    ##########################################################

    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)  # Currently random
    else:
        # Choose action with highest Q-value
        q_values = [Q_table[(state, a)] for a in range(n_actions)]
        return int(np.argmax(q_values))

    ##########################################################
    # END TODO
    ##########################################################


def update_Q(Q_table, state, action, reward, next_state, done, n_actions, alpha, gamma):
    """
    Update Q-table using Q-learning update rule

    Q(s,a) ← (1-α)Q(s,a) + α(r + γ max_a' Q(s',a'))  if not done
    Q(s,a) ← (1-α)Q(s,a) + αr                         if done

    Args:
        Q_table: Dictionary mapping (state, action) to Q-values
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode terminated
        n_actions: Number of available actions
        alpha: Learning rate
        gamma: Discount factor
    """
    ##########################################################
    # TODO: Implement Q-learning update
    # 1. Get current Q-value: Q_table[(state, action)]
    # 2. Compute target:
    #    - if done: target = reward
    #    - else: target = reward + gamma * max_a' Q(next_state, a')
    # 3. Update: Q(s,a) = (1-alpha) * Q(s,a) + alpha * target
    ##########################################################

    current_q = Q_table[(state, action)]

    if done:
        target = reward
    else:
        max_next_q = max(Q_table[(next_state, a)] for a in range(n_actions))
        target = reward + gamma * max_next_q

    Q_table[(state, action)] = (1 - alpha) * current_q + alpha * target

    ##########################################################
    # END TODO
    ##########################################################


if __name__ == "__main__":
    # Initialize environment
    env = gym.make("Taxi-v3", is_rainy=True)
    env.reset(seed=1)
    n_actions = env.action_space.n

    # Initialize Q-table
    Q_table = defaultdict(default_Q_value)

    # Training loop
    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done and step < MAX_STEPS:
            # Select action
            action = select_action(Q_table, state, n_actions, EPSILON)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Update Q-table
            update_Q(Q_table, state, action, reward, next_state, done,
                     n_actions, LEARNING_RATE, DISCOUNT_FACTOR)

            # Move to next state
            state = next_state
            step += 1

        # Decay epsilon
        EPSILON *= EPSILON_DECAY

        # Print progress
        if episode % 1000 == 0:
            print(f"Episode {episode}, Epsilon: {EPSILON:.3f}, Steps: {step}")

    # Save Q-table and final epsilon
    with open('Q_TABLE_QLearning.pkl', 'wb') as f:
        pickle.dump([Q_table, EPSILON], f)

    print("Training completed!")
