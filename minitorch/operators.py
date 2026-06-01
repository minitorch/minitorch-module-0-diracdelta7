"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from collections.abc import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Return the product of two numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Return the sum of two numbers."""
    return x + y


def neg(x: float) -> float:
    """Return the negation of the input."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if the first input is less than the second, else 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return whether two inputs are equal."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the larger of two inputs."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return whether two inputs differ by less than 1e-2."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Return the sigmoid of the input."""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def relu(x: float) -> float:
    """Return the ReLU of the input."""
    return x if x >= 0.0 else 0.0


def log(x: float) -> float:
    """Return the natural logarithm of the input."""
    return math.log(x)


def exp(x: float) -> float:
    """Return e raised to the input."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Return the backward value for the log function."""
    return y / x


def inv(x: float) -> float:
    """Return the multiplicative inverse of the input."""
    return 1 / x


def inv_back(x: float, y: float) -> float:
    """Return the backward value for the inverse function."""
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Return the backward value for the ReLU function."""
    if x > 0:
        return y
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], container: Iterable[float]) -> list[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [fn(num) for num in container]


def zipWith(
    fn: Callable[[float, float], float],
    container1: Iterable[float],
    container2: Iterable[float],
) -> list[float]:
    """Higher-order function that combines elements from two iterables using a given function."""
    return [fn(num1, num2) for num1, num2 in zip(container1, container2)]


def reduce(
    fn: Callable[[float, float], float], container: Iterable[float], start: float
) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function."""
    result = start
    for num in container:
        result = fn(result, num)
    return result


def negList(lst: list[float]) -> list[float]:
    """Negate a list."""
    return map(neg, lst)


def addLists(lst1: list[float], lst2: list[float]) -> list[float]:
    """Add two lists together."""
    return zipWith(add, lst1, lst2)


def sum(lst: list[float]) -> float:
    """Sum a list."""
    return reduce(add, lst, 0.0)


def prod(lst: list[float]) -> float:
    """Calculate the product of all elements in a list."""
    return reduce(mul, lst, 1.0)
