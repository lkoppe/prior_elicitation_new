# %%
import numpy as np
import scipy.stats
from scipy.stats import qmc
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

def generate_samples(n_samples: int, d: int = 1, method: str = "random"):
    """
    Generate samples using the specified method (quasi-random or random).

    Parameters:
    - n_samples (int): Number of samples to generate.
    - d (int): Dimensionality of the sample space (default: 1).
    - method (str): Sampling method, choose from 'random', 'sobol' or 'lhs' (default: 'random').

    Returns:
    - np.ndarray: Samples in the unit hypercube [0, 1]^d.
    """
    
    # Validate n_samples and d
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer.")
    
    # Validate method
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ["sobol", "lhs", "random"]:
        raise ValueError("Unsupported method. Choose from 'sobol', 'lhs', or 'random'.")

    # Initialize sample_data
    sample_data = None

    # Generate samples based on the chosen method
    if method == "sobol":
        sampler = qmc.Sobol(d=d)
        sample_data = sampler.random(n=n_samples)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=d)
        sample_data = sampler.random(n=n_samples)
    elif method == "random":
        sample_data = np.random.uniform(0, 1, size=(n_samples, d))
    
    return sample_data

def transform_samples_to_distribution(sample_data: np.ndarray, distributions):
    """
    Transform the sample_data to the specified distributions using the percent-point function (PPF).

    Parameters:
    - sample_data (np.ndarray): Samples in the unit hypercube [0, 1]^d, shape (n_samples, d_features).
    - distributions (list or object): A list of d distribution objects from scipy.stats or tfp.distributions
      (each having a .ppf/.quantile method), or a single distribution to apply to all columns. 

    Returns:
    - np.ndarray: Transformed sample_data to the specified distributions, shape (n_samples, d_features).

    Note:
    For a complete list of distributions available that can be used here
    (only univariate distributions) see
    - scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html 
    - tfp.distributions: https://www.tensorflow.org/probability/api_docs/python/tfp/distributions
    """
    
    # Validate distributions
    if not isinstance(distributions, (list, tuple)):
        distributions = [distributions] * sample_data.shape[1]  # Repeat the single distribution for each column

    if len(distributions) != sample_data.shape[1]:
        raise ValueError("The length of distributions must match the number of columns in samples.")
    
    # Initialize transformed samples array
    transformed_sample_data = np.empty_like(sample_data)

    # Transform each column to the specified distribution
    for i in range(sample_data.shape[1]):
        distribution = distributions[i]

        # Check if the distribution has either 'ppf' or 'quantile' method and use accordingly
        if hasattr(distribution, 'ppf'): 
            transformed_sample_data[:, i] = distribution.ppf(sample_data[:, i])
        elif hasattr(distribution, 'quantile'):
            transformed_sample_data[:, i] = distribution.quantile(sample_data[:, i])
        else:
            raise ValueError(f"Distribution {distribution} does not have 'ppf' or 'quantile' methods.")

    return transformed_sample_data

# %%
# Example
if __name__ == "__main__":
    # Generating some sample data in the unit hypercube
    samples = generate_samples(n_samples=2**6, d=3, method="sobol")  # 64 samples, 3 features/columns

    print("First 10 rows of hypercube samples:", end="\n\n")
    print(samples[:10,:], end="\n\n")

    # Define different distributions for each column
    distributions_list = [
        tfp.distributions.Normal(100,2),    # Normal distribution from tfp.distributions
        scipy.stats.binom(n=10, p=0.5),     # Binomial distribution from scipy.stats
        scipy.stats.expon()                 # Exponential distribution from scipy.stats
    ]

    # Transform samples to the specified distributions
    transformed_samples_list = transform_samples_to_distribution(samples, distributions_list)

    print("First 10 rows of distribution samples:", end="\n\n")
    print(transformed_samples_list[:10,:])
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(transformed_samples_list[:, 0], bins=20, alpha=0.5, color='blue', density=True)
    plt.title('Column 1 - Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Relative Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(transformed_samples_list[:, 1], bins=20, alpha=0.5, color='orange', density=True)
    plt.title('Column 2 - Binomial Distribution')
    plt.xlabel('Value')
    plt.ylabel('Relative Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(transformed_samples_list[:, 2], bins=20, alpha=0.5, color='green', density=True)
    plt.title('Column 3 - Exponential Distribution')
    plt.xlabel('Value')
    plt.ylabel('Relative Frequency')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
# %%
