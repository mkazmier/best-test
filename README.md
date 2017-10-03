# BEST test

## Description
Perform the Bayesian test for difference of means and standard deviations between parameters as described in the classic [BEST paper](http://www.indiana.edu/~kruschke/articles/Kruschke2013JEPG.pdf) (Kruschke, 2012). The prior distributions used are the same as in the paper (Student's T for data, normal for means, uniform for standard deviations, shifted exponential for normality)

## Example usage

```python
import numpy as np
import matplotlib.pyplot as plt
from best_test import BayesianDifferenceTest

observed_a = np.random.random(100)
observed_b = np.random.random(85) # unequal sample sizes are not an issue

mu_mean = 0 # mean for the prior on mean
mu_sd = 1 # standard deviation on the prior on mean
sd_lower = .1 # lower bound for the prior on standard deviation
sd_upper = 10 # upper bound for the prior on standard deviation
nu_mean = 30 # mean on the prior on nu (normality, aka 'degrees of freedom')

test = BayesianDifferenceTest('param_a_name', 'param_b_name',
                               mu_mean, mu_sd, 
                               sd_lower, sd_upper,
                               nu_mean)
test.run(observed_a, observed_b)
test.plot_posterior(ref_val=0)
plt.show()
```

## Notes
This program was written as a part of my internship in MAASTRO Clinic Knowledge Engineering team. It is available under the MIT licence.

## References
- Bayesian estimation supersedes the t test. Kruschke JK. J Exp Psychol Gen. 2013 May;142(2):573-603. doi: 10.1037/a0029146. Epub 2012 Jul 9.
