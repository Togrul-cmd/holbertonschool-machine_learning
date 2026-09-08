#!/usr/bin/env python3
"""
0-monte_carlo.py

Task 0 - Monte Carlo:
Performs first-visit Monte Carlo prediction to update the value
estimate V, using episodes sampled with `policy`.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env:        environment instance (gymnasium)
        V:          numpy.ndarray of shape (s,) containing the value estimate
        policy:     function that takes in a state and returns the next action
        episodes:   total number of episodes to train over
        max_steps:  maximum number of steps per episode
        alpha:      learning rate
        gamma:      discount rate

    Returns:
        V: the updated value estimate
    """
    V = np.asarray(V, dtype=np.float64)   # make sure we can store floats

    for _ in range(episodes):
        # ------------------------------------------------------------
        # 1) Sample one episode by following the policy
        # ------------------------------------------------------------
        state, _ = env.reset()
        episode = []                       # list of (state_t, reward_{t+1})

        for _ in range(max_steps):
            action = policy(state)         # a_t = policy(s_t)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = new_state
            if terminated or truncated:    # fell in hole / reached goal / timeout
                break

        # ------------------------------------------------------------
        # 2) Compute returns G_t backwards and update V
        #    (one update per distinct state -> first-visit MC)
        # ------------------------------------------------------------
        G = 0.0
        visited = set()
        for state, reward in reversed(episode):
            G = reward + gamma * G         # G_t = r_{t+1} + gamma * G_{t+1}
            if state not in visited:
                visited.add(state)
                V[state] += alpha * (G - V[state])

    return V