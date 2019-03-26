"""This module provides the gradient descent functionality."""

from collections import OrderedDict
from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import diff


class GradientDescent:

    def __init__(self, function, x: np.array, y: np.array, iterations: int = 100, learning_rate: int = 50):
        self.function = function
        self.function_type = type(function)
        self.x = x
        self.y = y
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.results = []

    def run(self):
        """Perform the gradient descent."""
        for i in range(self.iterations + 1):
            self.results.append({'step': i, 'cost': self.cost, 'variables': self.function.variables})

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

    def print_results(self, cost_precision: int = 5, variables_precision: int = 2):
        """Print the results of the gradient descent."""
        step_name, cost_name, variables_name = 'Step', 'Cost', 'Variables'
        digits = len(str(self.iterations))
        buffer = len(step_name) if len(step_name) > digits else digits

        print(f'{step_name:{buffer}} | {cost_name:{cost_precision + 6}} | {variables_name}')

        for row in self.results:
            variables = ' '.join([
                f'{name}={value:.{variables_precision}f}'
                for name, value in row['variables'].items()
            ])
            print(f"{row['step']:<{buffer}} | {row['cost']:.{cost_precision}E} | {variables}")

    def plot_cost_contours(self, steps: int = 100, vectors: bool = True):
        """Plot a 2-dimensional contour plot of cost versus the two function variables.

        The contours will be in the region of the minimum and maximum values of the function variables
        calculated during the gradient descent.
        """
        if len(self.function.get_variables) > 2:
            raise NotImplementedError("Cannot plot 2-dimensional plot with more than 2 variables.")
        if not self.results:
            raise ValueError('The gradient descent must be executed first.')

        variable_ranges = self._get_variable_ranges(steps=steps)

        costs = np.array([
            GradientDescent(self.function_type(*variables), self.x, self.y).cost
            for variables in product(*variable_ranges.values())
        ])

        costs = costs.reshape(*map(len, variable_ranges.values()))
        costs = costs.T
        var1, var2 = np.meshgrid(*variable_ranges.values())

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cs = ax.contour(var1, var2, costs)
        if vectors:
            ax.quiver(*self._get_vector_data(), angles='xy')
        var1_label, var2_label = variable_ranges.keys()
        plt.xlabel(var1_label)
        plt.ylabel(var2_label)
        plt.clabel(cs)
        plt.title(f'Contours of cost with {var1_label} and {var2_label}')
        plt.show()

    def _get_variable_values(self) -> OrderedDict:
        """Return all the values calculated for each variable.

        E.g.
        OrderedDict({
            'var1': [1.2, 1.1, 0.8, 0.3],
            'var2': [2.3, 3.1, 3.5, 3.5],
        })
        """
        return OrderedDict(
            {name: [result['variables'][name] for result in self.results] for name in self.function.variables.keys()}
        )

    def _get_variable_ranges(self, steps: int = 100) -> OrderedDict:
        """Return arrays of values for each variable between the minimum and maximum.

        E.g.
        OrderedDict({
            'var1': np.array([0.3, 0.4, ..., 1.2]),
            'var2': np.array([2.3, 3.1, ..., 3.5]),
        })
        """
        return OrderedDict(
            {name: np.linspace(min(values), max(values), steps) for name, values in self._get_variable_values().items()}
        )

    def _get_vector_data(self) -> Tuple[List, List, List, List]:

        var1_values, var2_values = self._get_variable_values().values()
        var1_diffs = diff(var1_values)
        var2_diffs = diff(var2_values)

        return var1_values[:-1], var2_values[:-1], var1_diffs, var2_diffs
