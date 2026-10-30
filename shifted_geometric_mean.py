import numpy as np


def geometric_mean(x):
    prod = np.prod(x)
    return prod ** (1 / len(x))


def shifted_geometric_mean(x, s):
    return geometric_mean(x - s) + s


def shifted_harmonic_mean(x, s):
    return len(x) / np.sum(1/(x - s)) +ｓ


def rank_weighted_min(x):
    weights = 1 / np.arange(1, len(x) + 1) ** 2
    weights = 1 / 2 ** np.arange(1, len(x) + 1)
    weights[-1] = weights[-2]
    return np.sum(weights * np.sort(x))


if __name__ == "__main__":
    sample = np.random.randint(0, 10, 5)
    print("sample:", sample)
    f1 = shifted_geometric_mean(sample, -1)
    print("shifted geometric mean:", f1)
    f2 = shifted_harmonic_mean(sample, -1)
    print("shifted harmonic mean:", f2)
    f3 = rank_weighted_min(sample)
    print("rank weighted min:", f3)
    minimum = np.min(sample)
    print("min:", minimum)

