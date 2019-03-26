# Gradient Descent

## Introduction

Gradient descent is the process of iteratively modifying
the variables of a function in a way that causes the error
between the output of the function and an existing answer to
be minimised for a particular input. We are travelling down
the 'error surface' by the steepest slope from where we started.  

## Functions

We will consider two functions to illustrate the idea: the
Normal distribution and a Linear function (or simply, a line).
Both of these functions have two variables that can be
manipulated in order to attempt to 'fit' the output of the 
functions onto some existing data. These variables are _sig_
(standard deviation) and _mu_ (average) for the Normal distribution
and _m_ (gradient) and _c_ (y-intercept) for a line.  

Our input data will be a vector of length 10 starting at -5
and ending at 5. 

We will create existing answers (or truths) using the functions,
the input data and values of _sig_ and _mu_ or _m_ and _c_. It is these
values of the variables that we expect our gradient descent algorithm
to converge towards from some initial values of _sig_ and _mu_ or 
_m_ and _c_.

The number of iterations and the learning rate are two further parameters
that will affect the success of the gradient descent and their effects will
be discussed. 

### Normal Distribution

```python
import numpy as np

from descent import GradientDescent
from functions import Normal

mu, sig = 1.0, 3.0
x = np.linspace(-5.0, 5.0, 10)
y = Normal(mu, sig).f(x)
mu_guess, sig_guess = 0.1, 3.8
model = Normal(mu_guess, sig_guess)
descent = GradientDescent(model, x, y, iterations=10, learning_rate=50)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_mu_sig_i10_r50]

![cost_contours_mu_sig_i10_r50]

#### Reduced Learning Rate

```python
descent = GradientDescent(model, x, y, iterations=10, learning_rate=5)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_mu_sig_i10_r5]

![cost_contours_mu_sig_i10_r5]

#### Increased Learning Rate

```python
descent = GradientDescent(model, x, y, iterations=10, learning_rate=120)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_mu_sig_i10_r120]

![cost_contours_mu_sig_i10_r120]

#### Increased Number of Iterations

```python
descent = GradientDescent(model, x, y, iterations=100, learning_rate=5)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_mu_sig_i100_r5]

![cost_contours_mu_sig_i100_r5]

### Linear Function

```python
from functions import Linear

m, c = 5.0, 3.0
x = np.linspace(-5.0, 5.0, 10)
y = Linear(m, c).f(x)

m_guess, c_guess = 0.1, 3.8
model = Linear(m_guess, c_guess)
descent = GradientDescent(model, x, y, iterations=1000, learning_rate=0.0001)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_m_c_i1000_r0.0001]

![cost_contours_m_c_i1000_r0.0001]

#### Reduced Learning Rate and Number of Iterations

```python
descent = GradientDescent(model, x, y, iterations=50, learning_rate=0.01)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_m_c_i50_r0.01]

![cost_contours_m_c_i50_r0.01]

## Conclusion

[cost_contours_mu_sig_i10_r5]: images/cost_contours_mu_sig_i10_r5.png "cost_contours_mu_sig_i10_r5"
[cost_contours_mu_sig_i10_r50]: images/cost_contours_mu_sig_i10_r50.png "cost_contours_mu_sig_i10_r50"
[cost_contours_mu_sig_i10_r120]: images/cost_contours_mu_sig_i10_r120.png "cost_contours_mu_sig_i10_r120"
[cost_contours_mu_sig_i100_r5]: images/cost_contours_mu_sig_i100_r5.png "cost_contours_mu_sig_i100_r5"
[cost_contours_m_c_i50_r0.01]: images/cost_contours_m_c_i50_r0.01.png "cost_contours_m_c_i50_r0.01"
[cost_contours_m_c_i1000_r0.0001]: images/cost_contours_m_c_i1000_r0.0001.png "cost_contours_m_c_i1000_r0.0001"
[results_mu_sig_i10_r5]: images/results_mu_sig_i10_r5.png "results_mu_sig_i10_r5"
[results_mu_sig_i10_r50]: images/results_mu_sig_i10_r50.png "results_mu_sig_i10_r50"
[results_mu_sig_i10_r120]: images/results_mu_sig_i10_r120.png "results_mu_sig_i10_r120"
[results_mu_sig_i100_r5]: images/results_mu_sig_i100_r5.png "results_mu_sig_i100_r5"
[results_m_c_i50_r0.01]: images/results_m_c_i50_r0.01.png "results_m_c_i50_r0.01"
[results_m_c_i1000_r0.0001]: images/results_m_c_i1000_r0.0001.png "results_m_c_i1000_r0.0001"
