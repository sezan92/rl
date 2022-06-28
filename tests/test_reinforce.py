import pytest
from rl.expected_reward import get_expected_reward, get_state_values


@pytest.mark.parametrize("rewards, expected",
[([1], [1]),
([0, 1], [0.99, 1]),
([1, 0], [1, 0]),
([0, 1, 1], [1.9701, 1.99, 1]),
([1, 0, 1], [1.9801, 0.99, 1]),
([0, 1, 0], [0.99, 1.0, 0]),
([1, 0, 1], [1.9801, 0.99, 1]),
([1, 0, 0], [1, 0, 0]),
([0, 0, 1], [0.9801, 0.99, 1])
])
def test_get_expected_reward(rewards, expected):
    assert get_expected_reward(rewards,0.99) == expected

# TODO: get_state_values