
import typing

import causalgraphicalmodels
import daft
import numpyro.diagnostics
import matplotlib.pyplot as plt

import arviz as az
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import numpyro
import numpyro.infer.autoguide
from numpyro.infer import Predictive

FittedParams = typing.Dict

def draw_dag(dag: causalgraphicalmodels.CausalGraphicalModel, coordinates: dict) -> None:
    assert not set(dag.dag.nodes) - coordinates.keys(), "All nodes should have defined coordinates"

    pgm = daft.PGM()
    for node in dag.dag.nodes:
        pgm.add_node(node, node, *coordinates[node])
    for edge in dag.dag.edges:
        pgm.add_edge(*edge)
    with plt.rc_context({"figure.constrained_layout.use": False}):
        pgm.render()
    plt.gca().invert_yaxis()


def print_filtered_summary(samples, prob=0.9, group_by_chain=True):
    filtered_samples = {name: values for name, values in samples.items() if len(values.shape) == 1}
    numpyro.diagnostics.print_summary(filtered_samples, prob=prob, group_by_chain=group_by_chain)


def plot_prior_predictive(guide: numpyro.infer.autoguide.AutoGuide, values: jnp.ndarray=jnp.array((-2, 2)), x_true=None, y_true=None, n_random_lines=0, ylim=(-2, 2)):
    predictive = Predictive(guide.model, num_samples=1000, return_sites=["mu"])
    prior_pred = predictive(jax.random.PRNGKey(10), jnp.array(values))
        
    mu = prior_pred["mu"]
    mu_mean = jnp.mean(mu, 0)
    mu_PI = jnp.percentile(mu, q=jnp.array((5.5, 94.5)), axis=0)
    
    if x_true is not None and y_true is not None:
        az.plot_pair({"x": x_true, "y": y_true})

    random_line_selection = mu[jax.random.randint(jax.random.PRNGKey(0), (n_random_lines,), 0, mu.shape[0]), :]
    for idx in range(n_random_lines):
        plt.plot(values, random_line_selection[idx, :], color="r", alpha=0.4)

    plt.plot(values, mu_mean, "k")
    plt.fill_between(values, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
    plt.ylim(ylim)
    plt.show()
        

def plot_posterior_predictive(guide: numpyro.infer.autoguide.AutoGuide, params: FittedParams, values: jnp.ndarray=jnp.array((-2, 2)), x_true=None, y_true=None):
    post = guide.sample_posterior(jax.random.PRNGKey(1), params, (1000,))
    post_pred = Predictive(guide.model, post)(jax.random.PRNGKey(2), values)

    mu = post_pred["mu"]
    mu_mean = jnp.mean(mu, 0)
    mu_PI = jnp.percentile(mu, q=jnp.array((5.5, 94.5)), axis=0)

    # plot it all
    if x_true is not None and y_true is not None:
        az.plot_pair({"x": x_true, "y": y_true})
    plt.plot(values, mu_mean, "k")
    plt.fill_between(values, mu_PI[0], mu_PI[1], color="k", alpha=0.2)
    plt.show()
