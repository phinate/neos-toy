from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxopt
import optax
from jax.random import multivariate_normal
from jax.random import PRNGKey

import pyhf


@partial(jax.jit, static_argnames=["model"])
def fisher_info_covariance(
    bestfit_pars: jnp.ndarray, model: pyhf.Model, observed_data: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.linalg.inv(
        jax.hessian(lambda lhood_pars: -model.logpdf(lhood_pars, observed_data)[0])(
            bestfit_pars,
        ),
    )


def gaussian_logpdf(
    bestfit_pars: jnp.ndarray, data: jnp.ndarray, cov: jnp.ndarray,
) -> jnp.ndarray:
    return jsp.stats.multivariate_normal.logpdf(data, bestfit_pars, cov).reshape(
        1,
    )


# @partial(jax.jit, static_argnames=["m"])
def model_gaussianity(
    model: pyhf.Model,
    bestfit_pars: jnp.ndarray,
    cov_approx: jnp.ndarray,
    observed_data: jnp.ndarray,
) -> jnp.ndarray:
    # - compare the likelihood of the fitted model with a gaussian approximation
    # that has the same MLE (fitted_pars)
    # - do this across a number of points in parspace (sampled from the gaussian approx)
    # and take the mean squared diff
    # - centre the values wrt the best-fit vals to scale the differences
    gaussian_parspace_samples = multivariate_normal(
        key=PRNGKey(1), mean=bestfit_pars, cov=cov_approx, shape=(100,),
    )

    relative_nlls_model = jax.vmap(
        lambda pars, data: -(
            model.logpdf(pars, data)[0] - model.logpdf(bestfit_pars, data)[0]
        ),  # scale origin to bestfit pars
        in_axes=(0, None),
    )(gaussian_parspace_samples, observed_data)

    relative_nlls_gaussian = jax.vmap(
        lambda pars, data: -(
            gaussian_logpdf(pars, data, cov_approx)[0]
            - gaussian_logpdf(bestfit_pars, data, cov_approx)[0]
        ),  # data fixes the lhood shape
        in_axes=(0, None),
    )(gaussian_parspace_samples, bestfit_pars)

    diffs = relative_nlls_model - relative_nlls_gaussian
    return jnp.mean(diffs[jnp.isfinite(diffs)] ** 2, axis=0)


def global_fit_objective(data: jnp.ndarray, model: pyhf.Model):
    def fit_objective(lhood_pars_to_optimize: jnp.ndarray) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        return -model.logpdf(lhood_pars_to_optimize, data)[
            0
        ]  # pyhf.Model.logpdf returns list[float]

    return fit_objective


def fixed_poi_fit_objective(data: jnp.ndarray, model: pyhf.Model, poi_condition: float):
    def fit_objective(lhood_pars_to_optimize: jnp.ndarray) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        poi_idx = model.config.poi_index
        pars = lhood_pars_to_optimize.at[poi_idx].set(poi_condition)
        # pyhf.Model.logpdf returns list[float]
        return -model.logpdf(pars, data)[0]

    return fit_objective


# try wrapping obj with closure_convert
@partial(jax.jit, static_argnames=["objective_fn"])  # forward pass
def _minimize(objective_fn, init_pars, lr):
    converted_fn, aux_pars = jax.closure_convert(objective_fn, init_pars)
    # aux_pars seems to be empty? took that line from jax docs example...
    solver = jaxopt.OptaxSolver(
        fun=converted_fn, opt=optax.adam(lr), implicit_diff=True,
    )
    return solver.run(init_pars, *aux_pars)[0]


# @partial(jax.jit, static_argnames=["model"]) # forward pass
def fit(
    data: jnp.ndarray,
    model: pyhf.Model,
    init_pars: jnp.ndarray,
    lr: float = 4e-3,
) -> jnp.ndarray:
    obj = global_fit_objective(data, model)
    # init_pars_copy = init_pars or jnp.asarray(model.config.suggested_init())
    fit_res = _minimize(obj, init_pars, lr)
    return fit_res


# @partial(jax.jit, static_argnames=["model"]) # forward pass
def fixed_poi_fit(
    data: jnp.ndarray,
    model: pyhf.Model,
    init_pars: jnp.ndarray,
    poi_condition: float,
    lr: float = 4e-3,
) -> jnp.ndarray:
    obj = fixed_poi_fit_objective(data, model, poi_condition)
    # init_pars_copy = init_pars or jnp.asarray(model.config.suggested_init())
    fit_res = _minimize(obj, init_pars, lr)
    blank = jnp.zeros_like(init_pars)
    blank += fit_res
    poi_idx = model.config.poi_index
    return blank.at[poi_idx].set(poi_condition)


# @partial(jax.jit, static_argnames=["model", "return_mle_pars"]) # forward pass
def hypotest(
    test_poi: float,
    data: jnp.ndarray,
    model: pyhf.Model,
    init_pars: jnp.ndarray,
    return_mle_pars: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:

    conditional_pars = fixed_poi_fit(
        data, model, poi_condition=test_poi, init_pars=init_pars,
    )
    mle_pars = fit(data, model, init_pars=init_pars)

    profile_likelihood = -2 * (
        model.logpdf(conditional_pars, data)[
            0
        ] - model.logpdf(mle_pars, data)[0]
    )

    poi_hat = mle_pars[model.config.poi_index]
    qmu = jnp.where(poi_hat < test_poi, profile_likelihood, 0.0)

    CLsb = 1 - pyhf.tensorlib.normal_cdf(jnp.sqrt(qmu))
    altval = 0.0
    CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
    CLs = CLsb / CLb
    return CLs, mle_pars if return_mle_pars else CLs
