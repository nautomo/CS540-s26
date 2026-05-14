import gymnasium as gym
import pickle
import numpy as np
import time


def default_Q_value():
    """Default Q-value for unseen state-action pairs"""
    return 0.0


def default_V_value():
    """Default V-value for unseen states"""
    return 0.0


def evaluate_Q_agent(Q_table, env_name, n_episodes=500, visualize=False):
    """
    Evaluate Q-learning agent performance with a greedy policy

    Args:
        Q_table: Dictionary mapping (state, action) to Q-values
        env_name: Name of the gym environment
        n_episodes: Number of episodes to evaluate
        visualize: Whether to render the environment

    Returns:
        Average reward over n_episodes
    """
    total_reward = 0
    env = gym.make(env_name, is_rainy=True, render_mode='human' if visualize else None)
    env.reset(seed=1)

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            q_values = [Q_table[(state, a)] for a in range(env.action_space.n)]
            action = np.argmax(q_values)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if visualize:
                time.sleep(0.05)

    env.close()
    avg_reward = total_reward / n_episodes
    return avg_reward


def evaluate_V_agent(V_table, env_name, n_episodes=500, visualize=False):
    """
    Evaluate Value Iteration agent performance

    Args:
        V_table: Dictionary mapping state to V-values
        env_name: Name of the gym environment
        n_episodes: Number of episodes to evaluate
        visualize: Whether to render the environment

    Returns:
        Average reward over n_episodes
    """
    total_reward = 0
    env = gym.make(env_name, is_rainy=True, render_mode='human' if visualize else None)
    env.reset(seed=1)

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            best_action = 0
            best_value = float('-inf')

            for action in range(env.action_space.n):
                expected_value = 0
                for prob, next_state, reward, terminated in env.unwrapped.P[state][action]:
                    expected_value += prob * (reward + 0.99 * V_table[next_state] * (1 - terminated))

                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action

            state, reward, terminated, truncated, _ = env.step(best_action)
            done = terminated or truncated
            total_reward += reward

            if visualize:
                time.sleep(0.05)

    env.close()
    avg_reward = total_reward / n_episodes
    return avg_reward


def test_agent(algo_name, visualize=False):
    """
    Test RL agent on Taxi-v3 and print results

    Args:
        algo_name: Algorithm name ('QLearning' or 'ValueIteration')
        visualize: Whether to render the environment
    """
    env_name = 'Taxi-v3'

    try:
        if algo_name == 'QLearning':
            with open('Q_TABLE_QLearning.pkl', 'rb') as f:
                Q_table, _ = pickle.load(f)
            score = evaluate_Q_agent(Q_table, env_name, visualize=visualize)

        elif algo_name == 'ValueIteration':
            with open('V_TABLE_ValueIteration.pkl', 'rb') as f:
                V_table = pickle.load(f)
            score = evaluate_V_agent(V_table, env_name, visualize=visualize)

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        print(f"{algo_name} on {env_name} (is_rainy=True):")
        print(f"Average episode reward over 500 episodes: {score:.4f}")
        print("Target score for full credit: >= 4.0")

        return score

    except FileNotFoundError as e:
        print(f"Error: Could not find saved model file.")
        print(f"Make sure you have run the training script and generated the .pkl file.")
        print(f"Details: {e}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("RL Agent Evaluation on Taxi-v3 (is_rainy=True)")
    print("=" * 60)

    # Test Q-Learning
    print("\n[Test 1] Q-Learning")
    print("-" * 60)
    test_agent('QLearning', visualize=False)

    # Test Value Iteration
    print("\n[Test 2] Value Iteration")
    print("-" * 60)
    test_agent('ValueIteration', visualize=False)

    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
