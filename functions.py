"""This module contains classes that represent general functions and
 which provide methods for their first partial derivatives.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
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
    def variables(self) -> OrderedDict:
        """Return the variable names of the function and their current values.

        The order should be the same as the order of the list of first partial derivatives of the variables.
        """
        pass

    @property
    @abstractmethod
    def get_variables(self) -> np.array:
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
        """Set the values of mean (mu) and standard deviation (sig) for the distribution."""
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
    def variables(self) -> OrderedDict:
        """Return the variable names and values."""
        return OrderedDict({'mu': self.mu, 'sig': self.sig})

    @property
    def get_variables(self) -> np.array:
        """Return the current values of the variables of the function."""
        return np.array([self.mu, self.sig])

    def set_variables(self, new_variables: np.array):
        """Set the values of the variables of the function."""
        self.mu, self.sig = new_variables


class Linear(AbstractFunction):
    """A Linear distribution"""

    def __init__(self, m, c):
        """Set the values of gradient (m) and intercept (c) for the line."""
        self.m = m
        self.c = c

    def f(self, x: np.array) -> np.array:
        """Return f(x; m, c)."""
        return self.m * x + self.c

    @staticmethod
    def dfdm(x: np.array) -> np.array:
        """Return d/dm of f(x; m, c)."""
        return x

    @staticmethod
    def dfdc(x: np.array) -> np.array:
        """Return d/dc of f(x; m, c)."""
        return np.ones(len(x))

    def get_partial_derivatives(self) -> List[Callable]:
        """Return the methods for each partial derivative."""
        return [self.dfdm, self.dfdc]

    @property
    def variables(self) -> OrderedDict:
        """Return the variable names and values."""
        return OrderedDict({'m': self.m, 'c': self.c})

    @property
    def get_variables(self) -> np.array:
        """Return the current values of the variables of the function."""
        return np.array([self.m, self.c])

    def set_variables(self, new_variables: np.array):
        """Set the values of the variables of the function."""
        self.m, self.c = new_variables
