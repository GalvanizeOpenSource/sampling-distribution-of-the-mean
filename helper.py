import numpy as np
import scipy.stats

class Binomial(object):

    def __init__(self, n, p):
        self._n = n
        self._p = p

    def sample(self, n):
        return np.random.binomial(n=self._n, p=self._p, size=n)

    def cdf(self, t):
        return scipy.stats.binom.cdf(t, self._n, self._p)

class Exponential(object):

    def __init__(self, a):
        self._a = a

    def sample(self, n):
        return np.random.exponential(scale=self._a, size=n)

    def cdf(self, t):
        return scipy.stats.expon.cdf(t, scale=(1/float(self._a)))

class Normal(object):

    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma

    def sample(self, n):
        return np.random.normal(loc=self._mu, scale=self._sigma, size=n)

    def cdf(self, t):
        return scipy.stats.norm.cdf(t, loc=self._mu, scale=self._sigma)

class Uniform(object):

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def sample(self, n):
        return np.random.uniform(low=self._a, high=self._b, size=n)

    def cdf(self, t):
        return scipy.stats.uniform.cdf(t, loc=self._a, scale=(self._b - self._a))

def scatterplot(data, ax, jitter=.125, c='black', **kwargs):
    ax.scatter(data, np.random.uniform(-jitter, jitter, size=len(data)), c=c, **kwargs)
    ax.yaxis.set_ticks([])

def sample_means(sample):
    sample = sample.astype(np.float64)
    cumsums = np.cumsum(sample)
    sample_sizes = np.arange(1, len(cumsums) + 1)
    sample_means = cumsums / sample_sizes
    return sample_sizes, sample_means

def sample_differences(sample):
    ones_and_negative_ones = sample + (sample - 1)
    sample_differences = np.cumsum(ones_and_negative_ones)
    sample_sizes = np.arange(1, len(sample_differences) + 1)
    return sample_sizes, sample_differences

def empirical_cdf(x, sample):
    cdf = np.zeros(shape=len(x))
    for data in sample:
        cdf += np.array(x >= data)
    return cdf / len(sample)

def dataset():
    return Normal(.1, .7).sample(50)

def density(data):
    return scipy.stats.kde.gaussian_kde(data)
