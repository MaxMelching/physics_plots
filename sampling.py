# -----------------------------------------------------------------------------
# 
# This script plots functions to sample an arbitrary probability density
# function (PDF) using the inverse transform method.
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/physics_plots
# 
# -----------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt



def get_cdf_vals(pdf: callable,
                 interval: any,
                 tolerance: float = 1e-5,
                 pdf_args: any = None
                 ) -> tuple[np.array]:
    """
    Computes the cumulative density function (CDF) for a given
    probability density function (PDF).

    Parameters
    ----------
    pdf : callable
        PDF for which the CDF values shall be computed. Must take point
        to evalute it in as first arguments. Any optional arguments can
        be given in the `pdf_args` parameter.
    interval : list or array-like
        Should contain values from the support of the pdf (where it is
        non-zero). It is ensured that this is the case up to a certain
        accuracy determined by the parameter `tolerance`.
    tolerance : float, optional, default = 1e-5
        Determines first and last point of `interval`. Values will be
        prepended or appended if corresponding CDF values are not inside
        allowed absolute deviation from 0 and 1, which is given by this
        parameter.
    pdf_args : any
        List or list-like which contains values to be used for optional
        parameters of `pdf` function.

    Returns
    -------
    A tuple of two numpy arrays. The first one contains points in which
    the CDF is evaluated. These evaluated values are given in the second
    array of the tuple.
    """

    step_size = interval[1] - interval[0]
    total_prob = pdf(interval[0], *pdf_args) * step_size  # Current value of the CDF

    # Ensure interval starts at zero probability (with certain tolerance)
    if total_prob > tolerance:
        max_appends = 0
        new_x_val = interval[0]

        interval = list(interval)  # Because appending more efficient here

        while total_prob > tolerance or max_appends > 1000:
            new_x_val -= step_size
            total_prob = pdf(new_x_val, *pdf_args) * step_size  # Integration step

            interval = [new_x_val] + interval  # Append at beginning

            max_appends += 1
    

    # Compute CDF values
    cdf = []  # List that will hold CDF values
    interval = np.array(interval)  # Important for speed, will have many accesses
    num_points = interval.size


    for i in range(num_points):
        if i < (num_points - 1):
            step_size = interval[i + 1] - interval[i]

        total_prob += pdf(interval[i], *pdf_args) * step_size
        cdf += [total_prob]


    # Ensure interval ends at probability 1 (with certain tolerance)
    if total_prob < (1 - tolerance):
        max_appends = 0
        index = num_points - 1

        new_x_val = interval[index]
        interval = list(interval)  # Because appending more efficient here

        while total_prob < (1 - tolerance) or max_appends > 1000:
            new_x_val += step_size

            interval += [new_x_val]
            total_prob += pdf(new_x_val, *pdf_args) * step_size  # Integration step
            cdf += [total_prob]

            index += 1
            max_appends += 1

    return np.array(interval), np.array(cdf)



def get_samples(N: int,
                cdf_vals: any,
                interval: any = None
                ) -> np.array:
    """
    Computes `N` samples for distribution with probability density
    function (PDF) that belongs to the cumulative density function (CDF)
    values given in `cdf_vals`. The sample values are taken from
    `interval`.

    A convenient way to get CDF values is the get_cdf_vals function.

    Parameters
    ----------
    N : int
        Number of samples to draw.
    cdf_vals : any
        Values of CDF at points in interval. Are assumed to go from 0
        to 1 (potentially with some numerical uncertainties).
    interval : any, optional, default = None
        Points of interval from which sample values are taken from. If
        None, a discrete PDF is assumed and it is taken to be the
        integers between 0, ..., length(cdf_vals).
    
    Returns
    -------
    samples : np.array
        Array containing N samples.
    """

    cdf_vals = np.array(cdf_vals)  # Important for speed, will have many accesses
    
    cdf_len = cdf_vals.size
    uniform_samples = np.random.random(N)  # Uniform random samples between 0, 1
    
    if interval is None:
        interval = np.array(range(cdf_len))
    else:
        interval = np.array(interval)  # Important for speed, will have many accesses
    
    assert cdf_len == interval.size


    # Compute samples
    samples = np.zeros(N)

    for i in range(N):
        if uniform_samples[i] < cdf_vals[0]:
            samples[i] = interval[0]
        for j in range(cdf_len - 1):
            if (cdf_vals[j] <= uniform_samples[i]) and (uniform_samples[i] < cdf_vals[j + 1]):
                samples[i] = interval[j + 1]
                break  # No need for further search for this sample
                
    return samples



if __name__ == '__main__':
    import math as m  # Needed for one of the distributions



    # ----- Gaussian distribution -----
    def gaussian(x, mu, sigma):
        return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    
    mu, sigma = 0, 1  # Parameters for standard normal distribution
    interval = np.arange(-5 * sigma, 5 * sigma, step=1e-3)

    interval, cdf_vals = get_cdf_vals(gaussian, interval, pdf_args=[mu, sigma])
    print('Computed Gaussian CDF values.')

    samples = get_samples(10_000, cdf_vals, interval)
    print('Got Gaussian samples.')


    plt.hist(samples, bins='doane', density=True, label='Histogram\nof Samples')
    plt.plot(interval, gaussian(interval, mu, sigma), label='Analytical\nDistribution')

    plt.title('Gaussian PDF and Samples')
    plt.legend()

    plt.show()



    # ----- Exponential distribution -----
    def exponential(x, k):
        return k * np.exp(- k * x)
    
    k = 3
    
    interval = np.linspace(0, 2.5)

    interval, cdf_vals = get_cdf_vals(exponential, interval, pdf_args=[k])
    print('Computed exponential CDF values.')

    samples = get_samples(10_000, cdf_vals, interval)
    print('Got exponential samples.')


    plt.hist(samples, bins='doane', density=True, label='Histogram\nof Samples')
    plt.plot(interval, exponential(interval, k), label='Analytical\nDistribution')

    plt.title('Exponential PDF and Samples')
    plt.legend()

    plt.show()



    # ----- Poisson distribution -----
    def poisson(n, k):
        return k ** n / m.factorial(n) * np.exp(- k)
        # return np.power(k, n) / m.factorial(n) * np.exp(- k)
    
    k = 4
    
    interval = np.array(range(12))

    interval, cdf_vals = get_cdf_vals(poisson, interval, pdf_args=[k])
    print('Computed poissonian CDF values.')

    samples = get_samples(10_000, cdf_vals, interval)
    print('Got poissonian samples.')


    plt.hist(samples, bins=interval, density=True, label='Histogram\nof Samples')
    # plt.plot(interval, poisson(interval, k), 'x', label='Analytical\nDistribution')
    plt.plot(interval, [poisson(x, k) for x in interval], 'x', label='Analytical\nDistribution')

    plt.title('Poissonian PDF and Samples')
    plt.legend()

    plt.show()