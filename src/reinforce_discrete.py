import argparse

from rl.policy import Policy
from rl.util import plot_scores, setup_environment, test_env
from rl.reinforce import reinforce_discrete
import torch

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)  # set random seed


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
        "--epsilon",
        type=float,
        default=0.9,
        help="Epsilon for exploration. Default: 0.9." + "Only if --train flag is set.",
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.999,
        help="Epsilon decay. Default: 0.999. Only if --train flag is set.",
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
            max_t=args.epoch,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
        )
        plot_scores(scores)
    elif args.infer:
        if args.infer_weight:
            policy.load_state_dict(torch.load(args.infer_weight))
            test_env(env, policy)
        else:
            raise ValueError("inference not given to --infer_weight.")
    else:
        raise ValueError("--train or --infer must be given.")
