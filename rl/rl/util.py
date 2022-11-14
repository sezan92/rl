import os

import cv2
import gym
import numpy as np
from matplotlib import pyplot as plt

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def setup_environment(env_name):
    env = gym.make(env_name)
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
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps,
        (frames[0].shape[0], frames[0].shape[1]),
    )
    for frame in frames:
        out.write(frame)
    out.release()


def test_env(
    env,
    policy,
    max_t=500,
    episode=5,
    render=False,
    video_path="",
    fps=10,
):
    rewards = []
    videos = []
    if video_path:
        os.makedirs(video_path, exist_ok=True)
        mode = "rgb_array"
    else:
        mode = "human"
    for e in range(episode):
        rendered_frames = []
        state = env.reset()
        sum_reward = 0
        for t in range(max_t):
            action, _ = policy.act(state)
            state, reward, done, _ = env.step(action)
            if render:
                rendered_frame = env.render(mode=mode)
                rendered_frames.append(rendered_frame)
            sum_reward += reward
            if done:
                break
        print(f"INFO: Episode: {e} , total reward: {sum_reward}")
        rewards.append(sum_reward)
        videos.append(rendered_frames)
    for i in range(videos):
        print(f"INFO: Saving video of episode: {i}")
        _write_video(
            rendered_frames,
            fps,
            os.path.join(video_path + f"episode_{i}_video.mp4"),
        )
