"""This module provides the gradient descent functionality."""

import numpy as np


class GradientDescent:

    def __init__(self, function, x: np.array, y: np.array, iterations: int = 100, learning_rate: int = 50):
        """"""
        self.function = function
        self.x = x
        self.y = y
        self.iterations = iterations
        self.learning_rate = learning_rate

    def run(self):
        for i in range(self.iterations):
            print(f'Step={i} Cost={self.cost()} Variables={vars(self.function)}')
            new_vars = old_vars * self.jacobian() * self.learning_rate

    def cost(self) -> float:
        """Return mod(y - f(x; mu, sig))**2."""
        return (self.y - self.function.f(self.x)).dot(self.y - self.function.f(self.x))

    def jacobian(self):
        """"""
        return [
            -2*(self.y - self.function.f(self.x)).dot(partial_derivative)
            for partial_derivative in self.function.first_partial_derivatives()
        ]
