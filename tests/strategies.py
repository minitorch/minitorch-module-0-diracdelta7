from hypothesis import settings
from hypothesis.strategies import floats, integers

import minitorch


settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
# We need this for the test of signoid's monotonicity, 100 will saturate float exp, which cases
# sigmoid(100) == sigmoid(100.001) == 1.0
moderate_floats = floats(min_value=-20, max_value=20, allow_nan=False)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def assert_close(a: float, b: float) -> None:
    assert minitorch.operators.is_close(a, b), f"Failure x={a} y={b}"
