import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict

# Hyperparameters
DISCOUNT_FACTOR = 0.99
THETA = 1e-8  # Convergence threshold
MAX_ITERATIONS = 10000


def default_V_value():
    """Default V-value for unseen states"""
    return 0.0


def value_iteration(env, gamma, theta, max_iterations):
    """
    Perform Value Iteration to compute optimal value function

    Args:
        env: Gymnasium environment
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        V_table: Dictionary mapping state to optimal V-values
    """
    # Initialize V-table
    V_table = defaultdict(default_V_value)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    for iteration in range(max_iterations):
        delta = 0

        ##########################################################
        # TODO: Implement Value Iteration
        #
        # For each state s:
        #   1. Compute action values for all actions:
        #      R(s,a) = Σ P(s'|s,a)[r + γ·V(s')]
        #   2. Update V(s) = max_a R(s,a)
        #   3. Track maximum change: delta = max(delta, |V_new(s) - V_old(s)|)
        #
        # Hint: Use env.unwrapped.P[state][action] to get transitions
        #       It returns list of (probability, next_state, reward, done)
        ##########################################################

        for state in range(n_states):
            for state in range(n_states):
                v_old = V_table[state]

                action_values = []

                for action in range(n_actions):
                    total = 0

                    for (prob, next_state, reward, done) in env.unwrapped.P[state][action]:
                        total += prob * (reward + gamma * V_table[next_state] * (1 - done))

                    action_values.append(total)

                V_table[state] = max(action_values)

                delta = max(delta, abs(v_old - V_table[state]))
                
        ##########################################################
        # END TODO
        ##########################################################

        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Delta: {delta:.6f}")

        # Check convergence
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break

    return V_table


if __name__ == "__main__":
    # Initialize environment
    env = gym.make("Taxi-v3", is_rainy=True)

    # Run Value Iteration
    print("Starting Value Iteration...")
    V_table = value_iteration(env, DISCOUNT_FACTOR, THETA, MAX_ITERATIONS)

    # Save V-table
    with open('V_TABLE_ValueIteration.pkl', 'wb') as f:
        pickle.dump(V_table, f)

    print("Training completed!")
    print(f"Number of states with non-zero values: {sum(1 for v in V_table.values() if v != 0)}")
