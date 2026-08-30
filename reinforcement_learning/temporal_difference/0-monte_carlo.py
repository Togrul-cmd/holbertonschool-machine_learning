import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: a function that takes in a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V, the updated value estimate
    """
    for _ in range(episodes):
        # Gymnasium API returns (state, info) on reset
        state, _ = env.reset()
        episode_data = []

        # 1. Generate an episode following the policy
        for _ in range(max_steps):
            action = policy(state)

            # Gymnasium API step returns 5 values
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        # 2. Calculate returns and update value estimates (First-visit MC)
        G = 0
        states_in_episode = [x[0] for x in episode_data]

        # Loop backwards through the episode to calculate the return G
        for i in reversed(range(len(episode_data))):
            state_t, reward_t1 = episode_data[i]

            # G = gamma * G + R_{t+1}
            G = gamma * G + reward_t1

            if state_t not in states_in_episode[:i]:
                # Update rule: V(s) = V(s) + alpha * [G - V(s)]
                V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V
