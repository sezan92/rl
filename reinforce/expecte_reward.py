# TODO: get expected reward, test if it is working
def get_expected_reward(states, rewards):
    expected_rewards = []
    for i in range(len(states)):
        expected_reward = 0
        gamma = 1
        for reward in rewards[i:]: 
            expected_reward += gamma * reward
            gamma = gamma * 0.99
        expected_rewards.append(expected_reward)
    return expected_rewards