import argparse

from rl.policy import Policy
from rl.util import plot_scores, setup_environment  # , test_env
from rl.reinforce import reinforce_discrete
from rl.util import test_env
from rl.util import moving_average
import numpy as np
import torch

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)  # set random seed
np.random.seed(RANDOM_SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        type=str,
        help="Environment name." + "Currently supported ['LunarLander-v2']",
    )
    parser.add_argument(
        "--train", action="store_true", help="Flag to" + " train or not"
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        help="Save the weights of the model.",
        default="reinforce.pth",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Flag to infer."
        + "If it is given, the path to the weight files must be given.",
    )
    parser.add_argument(
        "--infer_weight",
        type=str,
        help="Weight file to use for inference."
        + " Only used if --test flag is used. Default:None",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Discount factor for future rewards."
        + " Default: 0.8. Only if --train argument is given.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1000,
        help="Number of epochs. Default: 1000. Only if --train set is set.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate. Default: 0.001",
    )
    parser.add_argument(
        "--max_t", type=int, default=1000, help="maximum steps per episode"
    )
    parser.add_argument(
        "--plot_show",
        action="store_true",
        help="To plot the scores or not. Will work if --train is used.",
    )
    parser.add_argument(
        "--plot_fig_path",
        type=str,
        default="/plot.png",
        help="To save the plot of scores. Will work if --train is used. Default: /plot.png",
    )
    parser.add_argument(
        "--infer_video",
        type=str,
        default="",
        help="Save the inference performance in video. Only will work if --infer is used.",
    )
    parser.add_argument(
        "--infer_video_fps",
        type=int,
        default=10,
        help="Rendered video FPS. --infer must be given.",
    )
    parser.add_argument(
        "--infer_render", action="store_true", help="To render or not. Default: False."
    )

    args = parser.parse_args()
    env = setup_environment(args.env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = Policy(
        s_size=env.observation_space.shape[0], a_size=env.action_space.n
    ).to(device)
    if args.train:
        scores = reinforce_discrete(
            env,
            policy,
            args.save_model_path,
            gamma=args.gamma,
            n_episodes=args.epoch,
            max_t=args.max_t,
            learning_rate=args.learning_rate,
        )
        scores = moving_average(scores)
        plot_scores(scores, show=args.plot_show, plot_fig_path=args.plot_fig_path)
    elif args.infer:
        if args.infer_weight:
            policy.load_state_dict(torch.load(args.infer_weight))
            test_env(
                env,
                policy,
                render=args.infer_render,
                video_path=args.infer_video,
                fps=args.infer_video_fps,
            )
        else:
            raise ValueError("inference not given to --infer_weight.")
    else:
        raise ValueError("--train or --infer must be given.")
