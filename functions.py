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
        """Return f(x)."""
        pass

    @abstractmethod
    def get_partial_derivatives(self) -> List[Callable]:
        """Return the methods for each first partial derivative of the function.

        The order of the list should be the same as the order of the list of
        variables that the derivatives correspond to.
        """
        pass

    @property
    @abstractmethod
    def variables(self) -> np.array:
        """Return the variables of the function.

        The order of the list should be the same as the order of the list of
        first partial derivatives of the variables.
        """
        pass

    @abstractmethod
    def set_variables(self, new_variables: np.array):
        """Set the variables of the function.

        The order of the array should be the same as the order of variables
        specified on initialisation of the class instance.
        """
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
        return ((x - self.mu)/self.sig**2)*self.f(x)

    def dfdsig(self, x: np.array) -> np.array:
        """Return d/dsig of f(x; mu, sig)."""
        return (((x - self.mu)**2)/self.sig**2 - 1/self.sig)*self.f(x)

    def get_partial_derivatives(self) -> List[Callable]:
        """Return the methods for each partial derivative."""
        return [self.dfdmu, self.dfdsig]

    @property
    def variables(self) -> np.array:
        """Return the current values of the variables of the function."""
        return np.array([self.mu, self.sig])

    def set_variables(self, new_variables: np.array):
        """Set the values of the variables of the function."""
        self.mu, self.sig = new_variables
