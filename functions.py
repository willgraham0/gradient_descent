"""This module contains classes that represent general functions and
 which provide methods for their first partial derivatives.
"""

from abc import ABC, abstractmethod
from typing import Callable, List


import numpy as np


class AbstractFunction(ABC):
    """An abstract function that must be inherited by custom functions."""

    @abstractmethod
    def f(self, x: np.array) -> np.array:
        """Define the function and return the output data from it."""
        pass

    @abstractmethod
    def first_partial_derivatives(self) -> List[Callable]:
        """Return the methods for each first partial derivatives of the function."""
        pass


class Normal(AbstractFunction):
    """A Normal, or Gaussian, distribution."""

    def __init__(self, mu, sig):
        """Set the initial values of mu and sig for the distribution."""
        self.mu = mu
        self.sig = sig

    def f(self, x: np.array) -> np.array:
        """Return f(x; mu, sig)."""
        return (1/np.sqrt(2*np.pi*self.sig**2))*np.exp(-1*((x - self.mu)**2/(2*self.sig**2)))

    def dfdmu(self, x: np.array) -> np.array:
        """Return d/dmu of f(x; mu, sig)."""
        return ((x - self.mu)/self.sig**2) * self.f(x)

    def dfdsig(self, x: np.array) -> np.array:
        """Return d/dsig of f(x; mu, sig)."""
        return (((x - self.mu)**2)/self.sig**2 - 1/self.sig) * self.f(x)

    def first_partial_derivatives(self) -> List[Callable]:
        """Return the methods for each partial derivative."""
        return [self.dfdmu, self.dfdsig]

