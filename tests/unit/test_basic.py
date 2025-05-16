import pytest


@pytest.mark.unit
def test_one_plus_one_equals_two():
    assert 1 + 1 == 2


@pytest.mark.unit
def test_one_plus_two_equals_three():
    assert 1 + 2 == 3


@pytest.mark.unit
def test_two_plus_two_equals_four():
    assert 2 + 2 == 4


@pytest.mark.unit
def test_two_plus_three_equals_five():
    assert 2 + 3 == 5
