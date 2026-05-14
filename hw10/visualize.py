"""
Visualize learned policy from Q-Learning or Value Iteration on Taxi-v3
Shows the taxi's movement policy on the 5x5 grid for representative scenarios
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gymnasium as gym


def default_Q_value():
    return 0.0


def default_V_value():
    return 0.0


# Taxi-v3 state encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + dest
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
LOC_NAMES = ['R', 'G', 'Y', 'B']
ACTION_NAMES = {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Dropoff'}
ACTION_ARROWS = {0: '\u2193', 1: '\u2191', 2: '\u2192', 3: '\u2190', 4: 'P', 5: 'D'}


def encode_state(taxi_row, taxi_col, pass_loc, dest_idx):
    return ((taxi_row * 5 + taxi_col) * 5 + pass_loc) * 4 + dest_idx


def get_best_action_Q(Q_table, state, n_actions=6):
    q_values = [Q_table[(state, a)] for a in range(n_actions)]
    return np.argmax(q_values)


def get_best_action_V(V_table, env, state):
    best_action = 0
    best_value = float('-inf')
    for action in range(env.action_space.n):
        expected_value = 0
        for prob, next_state, reward, done in env.unwrapped.P[state][action]:
            expected_value += prob * (reward + 0.99 * V_table[next_state] * (1 - done))
        if expected_value > best_value:
            best_value = expected_value
            best_action = action
    return best_action


def plot_taxi_policy(get_action_fn, title, pass_loc, dest_idx):
    """
    Plot the taxi's movement policy on the 5x5 grid for a given
    passenger location and destination.

    Args:
        get_action_fn: function(taxi_row, taxi_col) -> action
        title: plot title
        pass_loc: passenger location index (0-3 for R/G/Y/B, 4 for in taxi)
        dest_idx: destination index (0-3 for R/G/Y/B)
    """
    n_rows, n_cols = 5, 5

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    for row in range(n_rows):
        for col in range(n_cols):
            y = n_rows - row - 1
            color = 'white'

            if (row, col) in LOCS:
                loc_idx = LOCS.index((row, col))
                if pass_loc < 4 and loc_idx == pass_loc:
                    color = '#FFD700'
                if loc_idx == dest_idx:
                    color = '#87CEEB'
                if pass_loc < 4 and loc_idx == pass_loc and loc_idx == dest_idx:
                    color = '#90EE90'

            rect = plt.Rectangle((col, y), 1, 1, facecolor=color,
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)

            action = get_action_fn(row, col)
            arrow = ACTION_ARROWS[action]
            fontsize = 28 if action < 4 else 22
            ax.text(col + 0.5, y + 0.5, arrow,
                   ha='center', va='center', fontsize=fontsize, fontweight='bold')

            if (row, col) in LOCS:
                loc_idx = LOCS.index((row, col))
                label = LOC_NAMES[loc_idx]
                ax.text(col + 0.5, y + 0.15, label,
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='darkred')

    # Draw walls (Taxi-v3 has walls between certain cells)
    wall_color = 'red'
    wall_width = 4
    ax.plot([2, 2], [n_rows - 0, n_rows - 2], color=wall_color, linewidth=wall_width)
    ax.plot([1, 1], [n_rows - 3, n_rows - 5], color=wall_color, linewidth=wall_width)
    ax.plot([3, 3], [n_rows - 3, n_rows - 5], color=wall_color, linewidth=wall_width)

    pass_str = LOC_NAMES[pass_loc] if pass_loc < 4 else "In Taxi"
    dest_str = LOC_NAMES[dest_idx]
    subtitle = f"Passenger: {pass_str}, Destination: {dest_str}"

    ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    legend_text = ("Yellow = Passenger pickup | Blue = Destination\n"
                   "Arrows = Move | P = Pickup | D = Dropoff")
    ax.text(2.5, -0.3, legend_text, ha='center', va='top', fontsize=10,
           style='italic', color='gray')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Policy Visualization for Taxi-v3")
    print("=" * 60)

    env = gym.make('Taxi-v3', is_rainy=True)

    # Scenario: Passenger at R(0,0), Destination B(4,3)
    pass_loc = 0  # R
    dest_idx = 3  # B

    # Visualize Q-Learning Policy
    try:
        print("\n[1] Loading Q-Learning policy...")
        Q_table, epsilon = pickle.load(open('Q_TABLE_QLearning.pkl', 'rb'))

        def q_action_pickup(r, c):
            return get_best_action_Q(Q_table, encode_state(r, c, pass_loc, dest_idx))

        def q_action_dropoff(r, c):
            return get_best_action_Q(Q_table, encode_state(r, c, 4, dest_idx))

        plot_taxi_policy(q_action_pickup, "Q-Learning: Navigate to Passenger",
                        pass_loc, dest_idx)
        plot_taxi_policy(q_action_dropoff, "Q-Learning: Deliver Passenger",
                        4, dest_idx)
    except FileNotFoundError:
        print("Error: Q_TABLE_QLearning.pkl not found")
    except Exception as e:
        print(f"Error loading Q-Learning policy: {e}")

    # Visualize Value Iteration Policy
    try:
        print("\n[2] Loading Value Iteration policy...")
        V_table = pickle.load(open('V_TABLE_ValueIteration.pkl', 'rb'))

        def v_action_pickup(r, c):
            return get_best_action_V(V_table, env, encode_state(r, c, pass_loc, dest_idx))

        def v_action_dropoff(r, c):
            return get_best_action_V(V_table, env, encode_state(r, c, 4, dest_idx))

        plot_taxi_policy(v_action_pickup, "Value Iteration: Navigate to Passenger",
                        pass_loc, dest_idx)
        plot_taxi_policy(v_action_dropoff, "Value Iteration: Deliver Passenger",
                        4, dest_idx)
    except FileNotFoundError:
        print("Error: V_TABLE_ValueIteration.pkl not found")
    except Exception as e:
        print(f"Error loading Value Iteration policy: {e}")
