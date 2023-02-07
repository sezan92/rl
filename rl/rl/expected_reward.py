# TODO: get expected reward, test if it is working
def get_expected_reward(rewards, gamma_init=1):
    expected_rewards = []
    for i in range(len(rewards)):
        expected_reward = 0
        gamma = 1
        for reward in rewards[i:]:
            expected_reward += gamma * reward
            gamma = gamma * gamma_init
        expected_rewards.append(expected_reward)
    return expected_rewards


def get_state_values(rewards):
    values = []
    for i in range(len(rewards)):
        values.append(sum(rewards[i:]) / len(rewards[i:]))
    return values
