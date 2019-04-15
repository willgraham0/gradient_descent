# Gradient Descent

## Introduction

Gradient descent is the process of iteratively modifying
the parameters of a function in a way that causes the error between the output of the function and an existing answer to
be minimised for a particular input. We travel down
the 'error surface' by the steepest slope from where we started.

## Functions

We will consider two functions to illustrate the idea: the
Normal distribution and the equation of a line.
Both of these functions have two parameters that can be
modified to attempt to 'fit' the output of the
functions onto an existing answer. These parameters are _mu_
(mean) and _sig_ (standard deviation) for the Normal distribution
and _m_ (gradient) and _c_ (y-intercept) for a line.

Our input data for both functions will be a vector of length 10 starting
at -5 and ending at 5.

We will create existing answers (or 'truths') using the functions,
the input data and values of _mu_ and _sig_ or _m_ and _c_. It is these
values of the parameters that we expect our gradient descent algorithm
to converge towards from some initial 'guess' values of _mu_ and _sig_ or
_m_ and _c_ that we will specify.

The number of iterations and the learning rate are two further parameters
that will affect the success of the gradient descent and their effects will
be discussed.

### Normal Distribution

First, we will create the 'truth' with values of _mu_ and _sig_ of 1.0 and 3.0,
respectively, and hold this vector in the variable _y_.

```python
import numpy as np

from descent import GradientDescent
from functions import Normal

mu, sig = 1.0, 3.0
x = np.linspace(-5.0, 5.0, 10)
y = Normal(mu, sig).f(x)
```

We make a first guess of _mu_ and _sig_ and run the gradient descent for
10 iterations and a learning rate of 50, printing the results and plotting a contour graph of constant cost (or error) with respect to the values of _mu_ and _sig_.

```python
mu_guess, sig_guess = 0.1, 3.8
model = Normal(mu_guess, sig_guess)
descent = GradientDescent(model, x, y, iterations=10, learning_rate=50)

descent.run()
descent.print_results()
```

Outputs:

```
Step | Cost        | Variables
0    | 4.98546E-03 | mu=0.10 sig=3.80
1    | 2.84884E-03 | mu=0.34 sig=3.54
2    | 1.18076E-03 | mu=0.56 sig=3.27
3    | 3.54710E-04 | mu=0.72 sig=3.07
4    | 1.08809E-04 | mu=0.84 sig=2.99
5    | 3.85440E-05 | mu=0.91 sig=2.97
6    | 1.40083E-05 | mu=0.95 sig=2.98
7    | 5.10998E-06 | mu=0.97 sig=2.99
8    | 1.86993E-06 | mu=0.98 sig=2.99
9    | 6.85950E-07 | mu=0.99 sig=2.99
10   | 2.52029E-07 | mu=0.99 sig=3.00
```

And:

```python
descent.plot_cost_contours()
```

Plots:

![cost_contours_mu_sig_i10_r50]

It can be seen from the results that the values of _mu_ and _sig_ are moving towards the true values of 1.0 and 3.0, respectively, after each iteration with the cost decreasing correspondingly. In the contour plot, this iteration is visualised by arrows progressing towards the minimum of the cost surface. The tail of the arrows indicates the current value of _mu_ and _sig_ and the
direction of the arrow indicates how the next values will be modified.

What the algorithm is doing is working out the _Jacobian_ of cost. This is a vector of the partial derivatives of the cost function with respect to the parameters _mu_ and _sig_ - a vector that points in the direction of maximum gradient (or change in cost). To minimise the cost, we want to the modify the values of _mu_ and _sig_ in a way that moves us in the direction on the _Jacobian_ and by an amount specified by the learning rate.


#### Reduced Learning Rate

We now investigate the effect of reducing the learning rate of the gradient descent. The learning rate is the size of the 'step' we are making as we descend the cost slope.

```python
# Reset the model
model = Normal(mu_guess, sig_guess)

descent = GradientDescent(model, x, y, iterations=10, learning_rate=5)
descent.run()
descent.print_results()
descent.plot_cost_contours()
```

Gives:

```
Step | Cost        | Variables
0    | 4.98546E-03 | mu=0.10 sig=3.80
1    | 4.76172E-03 | mu=0.12 sig=3.77
2    | 4.53924E-03 | mu=0.15 sig=3.75
3    | 4.31849E-03 | mu=0.17 sig=3.72
4    | 4.09999E-03 | mu=0.20 sig=3.69
5    | 3.88425E-03 | mu=0.22 sig=3.67
6    | 3.67182E-03 | mu=0.24 sig=3.64
7    | 3.46323E-03 | mu=0.27 sig=3.61
8    | 3.25903E-03 | mu=0.29 sig=3.59
9    | 3.05975E-03 | mu=0.31 sig=3.56
10   | 2.86590E-03 | mu=0.33 sig=3.53
```

And:

![cost_contours_mu_sig_i10_r5]

#### Increased Learning Rate

```python
# Reset the model
model = Normal(mu_guess, sig_guess)

descent = GradientDescent(model, x, y, iterations=10, learning_rate=120)
descent.run()
descent.print_results()
descent.plot_cost_contours()
```

Gives:

```
Step | Cost        | Variables
0    | 4.98546E-03 | mu=0.10 sig=3.80
1    | 5.96584E-04 | mu=0.69 sig=3.18
2    | 2.42676E-04 | mu=0.99 sig=2.81
3    | 2.73924E-04 | mu=0.98 sig=3.22
4    | 1.31326E-04 | mu=1.02 sig=2.86
5    | 1.53507E-04 | mu=0.98 sig=3.17
6    | 9.08440E-05 | mu=1.02 sig=2.88
7    | 1.02016E-04 | mu=0.98 sig=3.13
8    | 6.61375E-05 | mu=1.01 sig=2.90
9    | 7.22040E-05 | mu=0.99 sig=3.11
10   | 4.97031E-05 | mu=1.01 sig=2.91
```

And:

![cost_contours_mu_sig_i10_r120]

#### Increased Number of Iterations

```python
# Reset the model
model = Normal(mu_guess, sig_guess)

descent = GradientDescent(model, x, y, iterations=100, learning_rate=5)
descent.run()
descent.print_results()
descent.plot_cost_contours()
```

Gives:

```
Step | Cost        | Variables
0    | 4.98546E-03 | mu=0.10 sig=3.80
1    | 4.76172E-03 | mu=0.12 sig=3.77
2    | 4.53924E-03 | mu=0.15 sig=3.75
3    | 4.31849E-03 | mu=0.17 sig=3.72
...
97   | 1.65712E-06 | mu=0.98 sig=2.99
98   | 1.53095E-06 | mu=0.98 sig=2.99
99   | 1.41436E-06 | mu=0.98 sig=2.99
100  | 1.30663E-06 | mu=0.98 sig=2.99
```

And:

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

Gives:

```
Step | Cost        | Variables
0    | 2.45186E+03 | m=0.10 c=3.80
1    | 2.35322E+03 | m=0.20 c=3.80
2    | 2.25856E+03 | m=0.30 c=3.80
3    | 2.16771E+03 | m=0.39 c=3.80
...
997  | 1.18162E-01 | m=5.00 c=3.11
998  | 1.17690E-01 | m=5.00 c=3.11
999  | 1.17220E-01 | m=5.00 c=3.11
1000 | 1.16752E-01 | m=5.00 c=3.11
```

And:

![cost_contours_m_c_i1000_r0.0001]

In the plot above it appears that the path of the descent is cutting across the contours at an angle instead of perpendicular to them. The following plot, which has been adjusted to show contours of cost close to zero, reveals that the descent is indeed perpendicular to the contours.

![cost_contours_m_c_i1000_r0.0001_refined]

#### Reduced Learning Rate and Number of Iterations

```python
# Reset the model
model = Linear(m_guess, c_guess)
descent = GradientDescent(model, x, y, iterations=50, learning_rate=0.01)

descent.run()
descent.print_results()
descent.plot_cost_contours()
```

Gives:

```
Step | Cost        | Variables
0    | 2.45186E+03 | m=0.10 c=3.80
1    | 2.63406E+03 | m=10.08 c=3.64
2    | 2.83100E+03 | m=-0.27 c=3.51
3    | 3.04345E+03 | m=10.46 c=3.41
...
47   | 7.46487E+04 | m=32.07 c=3.00
48   | 8.02806E+04 | m=-23.08 c=3.00
49   | 8.63375E+04 | m=34.11 c=3.00
50   | 9.28513E+04 | m=-25.19 c=3.00
```

And:

![cost_contours_m_c_i50_r0.01]

## Conclusion

[cost_contours_mu_sig_i10_r5]: images/cost_contours_mu_sig_i10_r5.png "cost_contours_mu_sig_i10_r5"
[cost_contours_mu_sig_i10_r50]: images/cost_contours_mu_sig_i10_r50.png "cost_contours_mu_sig_i10_r50"
[cost_contours_mu_sig_i10_r120]: images/cost_contours_mu_sig_i10_r120.png "cost_contours_mu_sig_i10_r120"
[cost_contours_mu_sig_i100_r5]: images/cost_contours_mu_sig_i100_r5.png "cost_contours_mu_sig_i100_r5"
[cost_contours_m_c_i50_r0.01]: images/cost_contours_m_c_i50_r0.01.png "cost_contours_m_c_i50_r0.01"
[cost_contours_m_c_i1000_r0.0001_refined]:
images/cost_contours_m_c_i1000_r0.0001_refined.png "cost_contours_m_c_i1000_r0.0001_refined"
[cost_contours_m_c_i1000_r0.0001]:images/cost_contours_m_c_i1000_r0.0001.png "cost_contours_m_c_i1000_r0.0001"
