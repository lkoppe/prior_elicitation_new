import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

tfd = tfp.distributions
tfb = tfp.bijectors

class Normal_log():
    def __init__(self):
        self.name = "Normal_log_scale"
        self.parameters = ["loc","log_scale"]
    def __call__(self, loc, scale):
        """
        Instantiation of normal distribution with sigma being learned on the 
        log-scale.

        Parameters
        ----------
        loc : int
            location parameter of normal distribution.
        scale : int
            scale parameter of normal distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            normal distribution with sigma being on the log scale.

        """
        return tfd.Normal(loc, tf.exp(scale))
    
    
def custom_correlation(prior_samples):
    corM = tfp.stats.correlation(prior_samples, sample_axis=1, event_axis=-1)
    tensor = tf.experimental.numpy.triu(corM, 1)
    tensor_mask = tf.experimental.numpy.triu(corM, 1) != 0.

    cor = tf.boolean_mask(tensor, tensor_mask, axis=0)
    diag_elements = int((tensor.shape[-1]*(tensor.shape[-1]-1))/2)
    return tf.reshape(cor,(prior_samples.shape[0],diag_elements))

def custom_groups(ypred, gr):
    if gr == 1:
        return ypred[:,:,:14]
    if gr == 2:
        return ypred[:,:,14:36]
    if gr == 3:
        return ypred[:,:,36:]