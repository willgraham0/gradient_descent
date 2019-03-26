# Gradient Descent

## Introduction

Gradient descent is the process of iteratively modifying
the variables of a function in a way that causes the error
between the output of the function and an existing answer to
be minimised for a particular input - we are travelling down
the 'error surface' by the steepest slope.  

## Functions

We will consider two functions to illustrate the idea: the
Normal distribution and a Linear function (or simply a line).
Both of these functions have two variables that can be
manipulated in order to attempt to 'fit' the output of the 
function onto some existing data. 

### Normal Distribution

```python
import numpy as np

from descent import GradientDescent
from functions import Normal, Linear

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

```python
descent = GradientDescent(model, x, y, iterations=10, learning_rate=5)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_mu_sig_i10_r5]

![cost_contours_mu_sig_i10_r5]

```python
descent = GradientDescent(model, x, y, iterations=10, learning_rate=120)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

![results_mu_sig_i10_r120]

![cost_contours_mu_sig_i10_r120]

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
