import jax.numpy as jnp
from jax import lax, jit, partial, value_and_grad
from jax.experimental import optimizers
from util import progress_bar_scan, wait_until_computed


def build_optimiser(val_and_grad_loss, optimiser_type="adam"):
    @partial(jit, static_argnums=(0,))
    def run_optimiser(Niters, l_rate, x_data, y_data, params_IC):
    
        if optimiser_type == "sgd":
            opt_init, opt_update, get_params = optimizers.sgd(l_rate)
        elif optimiser_type == "adam":
            opt_init, opt_update, get_params = optimizers.adam(l_rate)
        else:
            raise ValueError("Optimiser not added.")

        @progress_bar_scan(Niters)
        def body(state, step):
            loss_val, loss_grad = val_and_grad_loss(get_params(state), x_data, y_data)
            state = opt_update(step, loss_grad, state)
            return state, loss_val

        state, loss_array = lax.scan(body, opt_init(params_IC), jnp.arange(Niters))
        return get_params(state), loss_array
    return run_optimiser
