import os

import cv2
import gym
import numpy as np
from matplotlib import pyplot as plt


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def setup_environment(env_name, mode):
    env = gym.make(env_name, render_mode=mode)
    return env


def plot_scores(scores, show=True, plot_fig_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    if show:
        plt.show()
    if plot_fig_path:
        print(f"INFO: Saving the plot in {plot_fig_path}")
        plt.savefig(plot_fig_path)
    return fig


def _write_video(frames, fps, output_path):
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out.release()


def test_env(
    env,
    policy,
    max_t=500,
    test_episode=5,
    render=False,
    video_path="",
    fps=10,
):
    rewards = []
    videos = []
    if video_path:
        os.makedirs(video_path, exist_ok=True)
    for e in range(test_episode):
        rendered_frames = []
        state, _ = env.reset()
        sum_reward = 0
        for t in range(max_t):
            action, _ = policy.act(state)
            state, reward, done, _, _ = env.step(action)
            if render:
                rendered_frame = env.render()
                rendered_frames.append(rendered_frame)
            sum_reward += reward
            if done:
                break
        print(f"INFO: Test Episode: {e} , total reward: {sum_reward}")
        rewards.append(sum_reward)
        videos.append(rendered_frames)
    for i in range(len(videos)):
        print(f"INFO: Saving video of test episode: {i}")
        _write_video(
            videos[i],
            fps,
            os.path.join(video_path, f"trial_{i}_video.avi"),
        )
