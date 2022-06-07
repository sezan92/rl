import argparse

import gym
from expected_reward import get_expected_reward, get_state_values

gym.logger.set_level(40) # suppress warnings (please remove if gives error)
from collections import deque

import numpy as np
import torch
from policy import Policy
from util import plot_scores

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED) # set random seed

import torch.optim as optim


def setup_environment(env_name):
    """
    Setup the environment
    """
    print(f'INFO: Setting up {env_name}')
    env = gym.make(env_name)
    env.seed(0)
    print(f'INFO: observation space: {env.observation_space}')
    print(f'INFO: action space: {env.action_space}')
    return env
    
def test_env(env, policy):
    """
    Test the environment with a given policy.
    """
    state = env.reset()
    for t in range(100000):
        action, _ = policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()

    env.close()

def reinforce(env, policy, model_weights_path, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, epsilon=0.1, epsilon_decay = 0.999, epsilon_min=0.1):
    scores_deque = deque(maxlen=100)
    scores = []
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        states = [state]
        for t in range(max_t):
            if np.random.random() > epsilon:
                action = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                log_prob = torch.from_numpy(np.array([np.log(0.25)]))
            else:
                action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        expected_rewards = get_expected_reward(rewards, gamma)
        state_values = get_state_values(rewards)
        
        policy_loss = []
        for i, log_prob in enumerate(saved_log_probs):
            A = expected_rewards[i] - state_values[i]
            policy_loss.append((-log_prob * A).float())
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if epsilon <= epsilon_min:
            epsilon = epsilon_min
        else:
            epsilon = epsilon * epsilon_decay
        
        if i_episode % print_every == 0:
            print('INFO: Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque)>=195.0:
            print('INFO: Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
    print(f'INFO: Saving the weights in {model_weights_path}')
    torch.save(policy.state_dict(), model_weights_path)
    return scores
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="Environment name. Currently supported ['LunarLander-v2']")
    parser.add_argument("--train", action="store_true", help="Flag to train or not")
    parser.add_argument("--save_model_path", type=str, help="Save the weights of the model.", default="reinforce.pth")
    parser.add_argument("--infer", action="store_true", help="Flag to infer. If it is given, the path to the weight files must be given.")
    parser.add_argument("--infer_weight", type=str, help="Weight file to use for inference. Only used if --test flag is used. Default:None")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor for future rewards. Default: 0.8. Only if --train argument is given.")
    parser.add_argument("--epoch", type=int, default=1000, help="Number of epochs. Default: 1000. Only if --train set is set.")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Epsilon for exploration. Default: 0.9. Only if --train flag is set.")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="Epsilon decay. Default: 0.999. Only if --train flag is set.")
    args = parser.parse_args()
    env = setup_environment(args.env)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = Policy(s_size=env.observation_space.shape[0], a_size=env.action_space.n).to(device)
    if args.train:
        scores = reinforce(env, policy, args.save_model_path, gamma=args.gamma, max_t=args.epoch, epsilon=args.epsilon, epsilon_decay=args.epsilon_decay)
        plot_scores(scores)
    elif args.infer:
        if args.infer_weight:
            policy.load_state_dict(torch.load(args.infer_weight))
            test_env(env, policy)
        else:
            raise ValueError('inference not given to --infer_weight.')
    else:
        raise ValueError('--train or --infer must be given.')