import numpy as np

from descent import GradientDescent
from functions import Normal, Linear


def main():
    mu, sig = 1.0, 3.0
    x = np.linspace(-5.0, 5.0, 10)
    y = Normal(mu, sig).f(x)

    mu_guess, sig_guess = 0.1, 3.8
    model = Normal(mu_guess, sig_guess)
    descent = GradientDescent(model, x, y, iterations=10, learning_rate=50)

    descent.run()
    descent.print_results()
    descent.plot_cost_contours(vectors=False)


    # m, c = 5.0, 3.0
    # x = np.linspace(-5.0, 5.0, 10)
    # y = Linear(m, c).f(x)
    #
    # m_guess, c_guess = 0.1, 3.8
    # model = Linear(m_guess, c_guess)
    # descent = GradientDescent(model, x, y, iterations=50, learning_rate=0.001)
    #
    # descent.run()
    # descent.print_results()
    # descent.plot_cost_contours(vectors=False)
