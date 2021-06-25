## General notes
Bayesian data analysis is no more than counting the numbers of ways the data could happen, according to our assumptions.
Statistical models work in two frames: the small world (the model itself) and the large world (where the model operates)
The inference of the model depends heavily on it's assumptions. Misleading models might lead to confident results.

Sampling is done to transform a calculus problem into a frequency problem. The integrals over probabilities are transformed to counting values.

## Technical notes
To get the value of a pdf in a certain point, use the exponential of the log_prob. E.g.:
`jnp.exp(dist.Binomial(total_count=9, probs=0.5).log_prob(6))`

To do quadratic approximation in numpyro, use `AutoLaplaceApproximation` and `SVI`:
```
def model(**variables):
    ...


guide = AutoLaplaceApproximation(model)
svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), **variables)
params, losses = svi.run(random.PRNGKey(0), 1000)

# display summary of quadratic approximation
samples = guide.sample_posterior(random.PRNGKey(1), params, (1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
```

## Definitions
Posterior distribution: For every unique combination of data, likelihood, parameters and prior, there is a unique posterior distribution. It contains the relative plausibility of the **parameter values**, conditional on the data and the model. It takes the form of the probability of the parameters, conditional on the data. The posterior is proportional to the product of the prior and the probability of the data.

Credible or compatibility intervals: the interval around the parameters that is most compatible with the data. There are two types: percentile intervals (PI), which leave out a certain percentage of both tails, and highest posterior density intervals (HDPI) which is the narrowest interval containing the specified mass. HDPI is more difficult to calculate and interpret.

Maximum A Posteriori (MAP) estimate: the value with the highest posterior probability

Posterior Predictive Distribution: for each combination of parameters, get a distribution of outcomes. Average these distributions with their respective likelihoods. When models become more complex, they can be used to understand the model by simulating implied observations.

A model has 3 elements:
- The variables, which can be observed (data) or ubonserved (parameters)
- All variables have a definition, either in terms of other variables or in terms of a probability distribution
- Combined, they form a joint generative model.

Prior predictive simulation: simulation from your generative model before data is added, to see the effect of modelling choices.

Conditional Independence: X is conditional independent of Y given Z if adding information on Y to a model of X and Z does not increase performance. Note that 

Markov Equivalence set: A set of DAGs with the same conditional independencies.

### Defining the model
#### Priors
- You can use different priors. See which one works best, and how sensitive the model is to the choice of prior before you decide.
- Priors cannot be based on values in the data!

#### Model itself
First, create a DAG of all variables. Capture those influence relationships in the model.
- use common and business sense
```py
def model(.., obs=None):
    # TODO
```

#### Plot the predictive prior
After defining a model, simulate the predictive prior to see the effects of the model for certain variables.
```py
def model(**variables):
    ...
    
guide = AutoLaplaceApproximation(model)

vars_to_check = [...]

predictive = Predictive(guide.model, num_samples=1000, return_sites=vars_to_check)
prior_pred = predictive(random.PRNGKey(10), **variables)

# use prior_pred to plot etc
```


### After fitting a model:
#### Summarizing the posterior
- How much posterior probability is below / above / between certain values
```
post_pred = Predictive(...)
posterior_mu = post_pred["mu"]
(mu < 0.5).mean(axis=0) if mu.ndim > 1 else (mu < 0.5).mean()
```
- Between which interval is a defined mass? I.e. **Credible or Compatibility interval**
```py
jnp.percentile(posterior_mu, q=(25, 75))
# or
numpyro.diagnostics.hpdi(posterior_mu, prob=0.5)
```

#### Use posterior prediction plots
Two reasons:
- Did the model correctly approximate the posterior (e.g. no computation / definition mistakes)
- How / where does the model fail?

```py
def model(**variables):
    ...
    
guide = AutoLaplaceApproximation(model)
svi = SVI(model, m5_3, optim.Adam(1), Trace_ELBO(), **variables)
params, losses = svi.run(random.PRNGKey(0), 1000)
posterior = guide.sample_posterior(random.PRNGKey(1), params, (1000,))

post_pred = Predictive(guide.model, posterior)(random.PRNGKey(2), **variables)
mu = post_pred["mu"]

# summarize samples across cases
mu_mean = jnp.mean(mu, 0)
mu_PI = jnp.percentile(mu, q=(5.5, 94.5), axis=0)

# simulate observations
# again no new data, so uses original data
y_sim = post_pred["y"]
y_PI = jnp.percentile(y_sim, q=(5.5, 94.5), axis=0)

ax = plt.subplot(
    ylim=(float(mu_PI.min()), float(mu_PI.max())),
    xlabel="Observed",
    ylabel="Predicted",
)
plt.plot(y, mu_mean, "o")
x = jnp.linspace(mu_PI.min(), mu_PI.max(), 101)
plt.plot(x, x, "--")
for i in range(y.shape[0]):
    # make line between two points where x is the same and y is the PI interval
    plt.plot([y[i]] * 2, mu_PI[:, i], "b")
fig = plt.gcf()

```


#### Fit counterfactual plots to what happens if you change variables
Three step approach:
1. Pick the variable to manipulate (intervention variable)
2. Define the range of values to set the intervention to
3. For each value in that range and for each sample in the posterior, use the causal model to simulate the values of other variables, including the outcome

```py
def model(**variables):
    ...
  
# first fit the model
guide = AutoLaplaceApproximation(model)
svi = SVI(model, quide, optim.Adam(0.1), Trace_ELBO(), **variables)
parameters, losses = svi.run(random.PRNGKey(0), 1000)

# then replace the variable of interest in the predictive
simulated_data = {"X": range(0, 10)}
posterior = guide.sample_posterior(random.PRNGKey(1), parameters, (1000,))
simulated_samples = Predictive(guide.model, posterior)(random.PRNGKey(2), **simulated_data)

```