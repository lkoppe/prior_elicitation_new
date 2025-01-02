# noqa SPDX-FileCopyrightText: 2024 Florence Bockting <florence.bockting@tu-dortmund.de>
#
# noqa SPDX-License-Identifier: Apache-2.0

import tensorflow as tf
import tensorflow_probability as tfp
import time
import numpy as np
import pandas as pd
import logging

from elicit.helper_functions import (
    save_as_pkl,
    save_hyperparameters,
    marginal_prior_moments,
)

tfd = tfp.distributions


def sgd_training(
    expert_elicited_statistics,
    prior_model_init,
    one_forward_simulation,
    compute_loss,
    global_dict,
    seed,
):
    """
    Wrapper that runs the optimization algorithms for E epochs.

    Parameters
    ----------
    expert_elicited_statistics : dict
        expert data or simulated data representing a prespecified ground truth.
    prior_model_init : class instance
        instance of a class that initializes and samples from the prior
        distributions.
    one_forward_simulation : callable
        one forward simulation cycle including: sampling from priors,
        simulating model
        predictions, computing target quantities and elicited statistics.
    compute_loss : callable
        sub-dag to compute the loss value including: compute loss components
        of model simulations and expert data, compute loss per component,
        compute total loss.
    global_dict : dict
        dictionary including all user-input settings.

    """
    # prepare generative model
    prior_model = prior_model_init
    total_loss = []
    component_losses = []
    gradients_ep = []
    time_per_epoch = []
    # create subdirectories for better readability
    dict_training = global_dict["training_settings"]
    dict_optimization = global_dict["optimization_settings"]
    # save initiale hyperparameter values (before any updating took place)
    if dict_training["method"] == "parametric_prior":
        res_dict = save_hyperparameters(prior_model_init, 0, global_dict)
    for epoch in tf.range(dict_training["epochs"]):
        if epoch > 0:
            logging.disable(logging.INFO)
        # runtime of one epoch
        epoch_time_start = time.time()
        # initialize the adam optimizer
        get_optimizer = dict_optimization["optimizer"]
        args_optimizer = dict_optimization["optimizer_specs"]
        optimizer = get_optimizer(**args_optimizer)
        with tf.GradientTape() as tape:
            # generate simulations from model
            training_elicited_statistics = one_forward_simulation(
                prior_model, global_dict,
            )
            # comput loss
            weighted_total_loss = compute_loss(
                training_elicited_statistics,
                expert_elicited_statistics,
                global_dict,
                epoch,
            )
            # compute gradient of loss wrt trainable_variables
            gradients = tape.gradient(
                weighted_total_loss, prior_model.trainable_variables
            )
            # update trainable_variables using gradient info with adam
            # optimizer
            optimizer.apply_gradients(
                zip(gradients, prior_model.trainable_variables)
                )

        # time end of epoch
        epoch_time_end = time.time()
        epoch_time = epoch_time_end - epoch_time_start

        # break for loop if loss is NAN and inform about cause
        if tf.math.is_nan(weighted_total_loss):
            print("Loss is NAN. The training process has been stopped.")
            break

        if dict_training["progress_info"] > 0:
            # print information for user during training
            # inform about epoch time, total loss and learning rate
            if epoch % dict_training["view_ep"] == 0:
                if (
                    type(
                        dict_optimization["optimizer_specs"][
                            "learning_rate"
                        ]
                    )
                    is float
                ):
                    lr = dict_optimization["optimizer_specs"][
                        "learning_rate"
                    ]
                else:
                    lr = dict_optimization["optimizer_specs"][
                        "learning_rate"
                    ](epoch)
                print(f"epoch_time: {epoch_time:.3f} sec")
                print(f"Epoch: {epoch}, loss: {weighted_total_loss:.5f},\
                      lr: {lr:.6f}")
            # inform about estimated time until completion
            if epoch > 0 and epoch % dict_training["view_ep"] == 0:
                avg_ep_time = np.mean(time_per_epoch)
                remaining_eps = np.subtract(
                    dict_training["epochs"], epoch
                )
                estimated_time = np.multiply(remaining_eps, avg_ep_time)
                timing = time.strftime('%H:%M:%S', time.gmtime(estimated_time))
                print(f"Estimated time until completion: {timing}")

        if epoch == np.subtract(dict_training["epochs"], 1):
            print("Done :)")

        # save gradients in file
        saving_path = global_dict["training_settings"]["output_path"]
        if dict_training["method"] == "parametric_prior":
            path = saving_path + "/gradients.pkl"
            save_as_pkl(gradients, path)
            # save for each epoch
            gradients_ep.append(gradients)

        # savings per epoch
        time_per_epoch.append(epoch_time)
        total_loss.append(weighted_total_loss)
        component_losses.append(
            pd.read_pickle(saving_path + "/loss_per_component.pkl")
            )

        if dict_training["method"] == "parametric_prior":
            # save single learned hyperparameter values for each prior and
            # epoch
            res_dict = save_hyperparameters(prior_model, 1, global_dict)
        else:
            # save mean and std for each sampled marginal prior; for each epoch
            path_model = saving_path + "/model_simulations.pkl"
            res_dict = marginal_prior_moments(
                pd.read_pickle(path_model)["prior_samples"],
                epoch,
                global_dict,
            )

        if epoch == 0:
            train_var = prior_model.trainable_variables
            # save final results in file
    res = {
        "loss": total_loss,
        "loss_component": component_losses,
        "hyperparameter": res_dict,
        "time_epoch": time_per_epoch,
        "seed": seed,
        "nf_weights": train_var
    }

    if dict_training["method"] == "parametric_prior":
        res["gradients"] = gradients_ep
    path = saving_path + "/final_results.pkl"
    save_as_pkl(res, path)
