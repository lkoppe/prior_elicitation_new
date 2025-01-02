# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import logging
import elicit.logs_config # noqa

from elicit.helper_functions import save_as_pkl # noqa

tfd = tfp.distributions


def softmax_gumbel_trick(model_simulations, global_dict):
    """
    The softmax-gumbel trick computes a continuous approximation of ypred from
    a discrete likelihood and thus allows for the computation of gradients for
    discrete random variables.

    Currently this approach is only implemented for models without upper
    boundary (e.g., Poisson model).

    Corresponding literature:

    - Maddison, C. J., Mnih, A. & Teh, Y. W. The concrete distribution:
        A continuous relaxation of
      discrete random variables in International Conference on Learning
      Representations (2017). https://doi.org/10.48550/arXiv.1611.00712
    - Jang, E., Gu, S. & Poole, B. Categorical reparameterization with
    gumbel-softmax in International Conference on Learning Representations
    (2017). https://openreview.net/forum?id=rkE3y85ee.
    - Joo, W., Kim, D., Shin, S. & Moon, I.-C. Generalized gumbel-softmax
    gradient estimator for generic discrete random variables.
      Preprint at https://doi.org/10.48550/arXiv.2003.01847 (2020).

    Parameters
    ----------
    model_simulations : dict
        dictionary containing all simulated output variables from the
        generative model.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    ypred : tf.Tensor
        continuously approximated ypred from the discrete likelihood.

    """
    # set seed
    tf.random.set_seed(global_dict["training_settings"]["seed"])
    # get batch size
    B = model_simulations["epred"].shape[0]
    # get number of simulations from priors
    S = model_simulations["epred"].shape[1]
    # get number of observations
    number_obs = model_simulations["epred"].shape[2]
    # create subdictionaries for better readability
    dict_generator = global_dict["generative_model"]
    # constant outcome vector (including zero outcome)
    thres = dict_generator["softmax_gumble_specs"]["upper_threshold"]
    c = tf.range(thres + 1, delta=1, dtype=tf.float32)
    # broadcast to shape (B, rep, outcome-length)
    c_brct = tf.broadcast_to(c[None, None, None, :], shape=(B, S, number_obs,
                                                            len(c)))
    # compute pmf value
    pi = model_simulations["likelihood"].prob(c_brct)
    # prevent underflow
    pi = tf.where(pi < 1.8 * 10 ** (-30), 1.8 * 10 ** (-30), pi)
    # sample from uniform
    u = tfd.Uniform(0, 1).sample((B, S, number_obs, len(c)))
    # generate a gumbel sample from uniform sample
    g = -tf.math.log(-tf.math.log(u))
    # softmax gumbel trick
    w = tf.nn.softmax(
        tf.math.divide(
            tf.math.add(tf.math.log(pi), g),
            dict_generator["softmax_gumble_specs"]["temperature"],
        )
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    return ypred


def simulate_from_generator(prior_samples, ground_truth, global_dict):
    """
    Simulates data from the specified generative model.

    Parameters
    ----------
    prior_samples : dict
        samples from prior distributions.
    ground_truth : bool
        if simulation is based on true hyperparameter vector. Mainly for
        saving results in a specific "expert" folder for later analysis.
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    model_simulations : dict
        simulated data from generative model.

    """
    logger = logging.getLogger(__name__)
    if ground_truth:
        logger.info("simulate from true generative model")
    else:
        logger.info("simulate from generative model")

    # set seed
    tf.random.set_seed(global_dict["training_settings"]["seed"])
    # create subdictionaries for better readability
    dict_generator = global_dict["generative_model"]
    # get model and initialize generative model
    GenerativeModel = dict_generator["model"]
    generative_model = GenerativeModel()
    # get model specific arguments (that are not prior samples)
    add_model_args = dict_generator["additional_model_args"]
    # simulate from generator
    if add_model_args is not None:
        model_simulations = generative_model(
            ground_truth, prior_samples, **add_model_args
        )
    else:
        model_simulations = generative_model(ground_truth, prior_samples)

    # estimate gradients for discrete likelihood if necessary
    if dict_generator["discrete_likelihood"]:
        model_simulations["ypred"] = softmax_gumbel_trick(
            model_simulations, global_dict
        )
    # save file in object
    saving_path = global_dict["training_settings"]["output_path"]
    if ground_truth:
        saving_path = saving_path + "/expert"
    path = saving_path + "/model_simulations.pkl"
    save_as_pkl(model_simulations, path)

    return model_simulations
