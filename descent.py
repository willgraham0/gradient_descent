"""This module provides the gradient descent functionality."""

import numpy as np


class GradientDescent:

    def __init__(
            self,
            function,
            x: np.array,
            y: np.array,
            iterations: int = 100,
            learning_rate: int = 50):
        self.function = function
        self.x = x
        self.y = y
        self.iterations = iterations
        self.learning_rate = learning_rate

    def run(self):
        """Perform the gradient descent."""
        for i in range(self.iterations):
            print(f'Step={i:{len(str(self.iterations))}} Cost={self.cost:.5E} Variables={vars(self.function)}')
            new_variables = self.function.variables - self.jacobian * self.learning_rate
            self.function.set_variables(new_variables)

    @property
    def cost(self) -> float:
        """Return mod(y - f(x; variables))**2."""
        difference_vector = self.y - self.function.f(self.x)
        return difference_vector.dot(difference_vector)

    @property
    def jacobian(self) -> np.array:
        """Return the Jacobian (vector of first derivative of each variable)."""
        return np.array([
            -2 * (self.y - self.function.f(self.x)).dot(partial_derivative(self.x))
            for partial_derivative in self.function.get_partial_derivatives()
        ])
