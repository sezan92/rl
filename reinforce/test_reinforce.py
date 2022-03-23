import pytest
from expecte_reward import get_expected_reward

@pytest.mark.parametrize("states, rewards, expected",
[([0], [1], [1]),
([0, 1], [0, 1], [0.99, 1]),
([0, 1], [1, 0], [1, 0]),
([0, 1, 2], [0, 1, 1], [1.9701, 1.99, 1]),
([0, 1, 2], [1, 0, 1], [1.9801, 0.99, 1]),
([0, 1, 2], [0, 1, 0], [0.99, 1.0, 0]),
([0, 1, 2], [1, 0, 1], [1.9801, 0.99, 1]),
([0, 1, 2], [1, 0, 0], [1, 0, 0]),
([0, 1, 2], [0, 0, 1], [0.9801, 0.99, 1])
])
def test_get_expected_reward(states, rewards, expected):
    assert get_expected_reward(states, rewards) == expected
