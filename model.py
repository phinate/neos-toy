from __future__ import annotations

import jax.numpy as jnp

import pyhf


def make_model(pars: jnp.ndarray) -> pyhf.Model:
    bounded_pars_upper = jnp.where(pars > 10.0, 10.0, pars)
    bounded_pars = jnp.where(
        bounded_pars_upper < -
        10.0, -10.0, bounded_pars_upper,
    )

    u1, d1 = bounded_pars
    u = jnp.array([u1, -u1])
    d = jnp.array([d1, -d1])

    sig = jnp.array([1, 9])
    nominal = jnp.array([50, 50])
    up = jnp.array([50, 50]) + u
    down = jnp.array([50, 50]) + d

    m = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": sig,
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None},
                        ],
                    },
                    {
                        "name": "background",
                        "data": nominal,
                        "modifiers": [
                            {
                                "name": "bkguncrt",
                                "type": "histosys",
                                "data": {"hi_data": up, "lo_data": down},
                            },
                        ],
                    },
                ],
            },
        ],
    }
    return pyhf.Model(m, validate=False)
