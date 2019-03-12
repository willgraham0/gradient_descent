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
        """Perform the gradient descent and print progress to console."""
        step_name = 'Step'
        digits = len(str(self.iterations))
        buffer = len(step_name) if len(step_name) > digits else digits
        cost_name, cost_precision = 'Cost', 5
        variables_name, variables_precision = 'Variables', 2

        print(
            f'{step_name:{buffer}} | '
            f'{cost_name:{cost_precision + 6}} | '
            f'{variables_name}'
        )

        for i in range(self.iterations):
            variables = ' '.join(
                [f'{name}={value:.{variables_precision}f}' for name, value in vars(self.function).items()]
            )
            print(f'{i:<{buffer}} | {self.cost:.{cost_precision}E} | {variables}')

            # Increment function variables in the direction of largest cost reduction
            new_variables = self.function.get_variables - self.jacobian * self.learning_rate
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
