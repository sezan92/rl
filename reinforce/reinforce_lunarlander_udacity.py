import argparse
from expected_reward import get_expected_reward, get_state_values
import gym
gym.logger.set_level(40) # suppress warnings (please remove if gives error)
import numpy as np
from collections import deque
from util import plot_scores
from policy import Policy
import torch
torch.manual_seed(0) # set random seed

import torch.optim as optim

ENV_NAME = "LunarLander-v2"

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


policy = Policy(s_size=env.observation_space.shape[0], a_size=env.action_space.n).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def reinforce(env, model_weights_path, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, epsilon=0.1, epsilon_decay = 0.999, epsilon_min=0.1):
    scores_deque = deque(maxlen=100)
    scores = []
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
            policy_loss.append(-log_prob * A)
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
    parser = argparse.ArgumentDefaultsHelpFormatter()
    parser.add_argument("env", type=str, help="Environment name. Currently supported ['LunarLander-v2']")
    parser.add_argument("--train", action_store=True, help="Flag to train or play")
    parser.add_argument("--save_model_path", type=str, help="Save the weights of the model.", default="reinforce.pth")

    args = parser.parse_args()
    env = setup_environment(args.env)
    if args.train:
        scores = reinforce(env, args.save_model_path, gamma=0.8, max_t=10000, epsilon=0.9, epsilon_decay=0.999)
        plot_scores(scores)
        # TODO: test the script asap.