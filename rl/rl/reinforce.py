import os

from rl.expected_reward import get_expected_reward, get_state_values
from rl.util import test_env

import numpy as np
import torch
import torch.optim as optim


def reinforce_discrete(
    env,
    policy,
    model_weights_path,
    n_episodes=1000,
    max_t=1000,
    gamma=1.0,
    print_every=100,
    learning_rate=1e-2,
    video_every="",
    video_dir=None,
    fps=20,
):
    scores = []
    if (video_dir and not video_every) or (video_every and not video_dir):
        raise ValueError("Both video_path and video_every argument are needed.")
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        states = [state]
        for _ in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)

            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            if terminated or truncated:
                break
        scores.append(sum(rewards))

        expected_rewards = get_expected_reward(rewards, gamma)
        state_values = get_state_values(rewards)

        policy_loss = []
        for i, log_prob in enumerate(saved_log_probs):
            A = expected_rewards[i] - np.mean(expected_rewards)
            policy_loss.append((-log_prob * A).float())
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                "INFO: Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores[-print_every:])
                )
            )
        if np.mean(scores[-print_every:]) >= 195.0:
            print(
                "INFO: Environment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores[-print_every:])
                )
            )
            break
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
            if i_episode % video_every == 0:
                video_path = os.path.join(video_dir, f"{i_episode}")
                test_env(env, policy, render=True, video_path=video_path, fps=fps)
    print(f"INFO: Saving the weights in {model_weights_path}")

    torch.save(policy.state_dict(), model_weights_path)
    return scores
