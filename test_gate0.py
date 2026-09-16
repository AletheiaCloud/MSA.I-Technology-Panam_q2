import numpy as np
import pytest

def test_environment_is_ready():
    """Sprawdza, czy pytest i numpy działają w izolowanym środowisku."""
    assert True

def test_numpy_complex_numbers():
    """Sprawdza, czy numpy obsługuje liczby zespolone (wymagane do stanów kwantowych)."""
    state = np.array([1+0j, 0+0j])
    assert np.isclose(np.abs(state[0]), 1.0)