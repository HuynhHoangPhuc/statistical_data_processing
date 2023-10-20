from scipy.stats import t


def describe(arr: list) -> tuple:
    n = len(arr)
    mean = sum(arr) / n
    var = sum([(e - mean) ** 2 for e in arr]) / (n - 1)
    std = var ** 0.5
    minimum = min(arr)
    maximum = max(arr)
    median = arr[n // 2 + 1] if n & 1 else (arr[n // 2] + arr[n // 2]) / 2

    return mean, var, std, minimum, maximum, median


def cut(arr: list, bin_size: int) -> dict[str, int]:
    _, _, _, minimum, maximum, _ = describe(arr)
    interval = [minimum - bin_size / 2, minimum + bin_size / 2]
    frequency = []
    while interval[-1] <= maximum:
        frequency.append(0)
        for e in arr:
            if interval[-2] <= e < interval[-1]:
                frequency[-1] += 1
        interval.append(interval[-1] + bin_size)
    frequency.append(len(arr) - sum(frequency))

    label = []
    for i in range(len(interval) - 1):
        label.append(f'[{interval[i]}, {interval[i + 1]})')

    return dict(zip(label, frequency))


def quantile(arr: list, q: float) -> float:
    return sorted(arr)[int(len(arr) * q)]


def statistics(arr: list, m: float, kind: int, alpha: float) -> tuple:
    n = len(arr)
    mean, var, _, _, _, _ = describe(arr)
    t0 = (mean - m) / (var * (n ** 0.5))
    e = t(n - 1).ppf(1 - alpha / 2)

    if kind == 0:
        p = 2 * (1 - t(n - 1).cdf(abs(t0)))
    elif kind == 1:
        p = t(n - 1).cdf(t0)
    else:
        p = 1 - t(n - 1).cdf(t0)

    return mean, p, kind, [mean - e, mean + e]


if __name__ == '__main__':
    data = [94.1, 86.1, 95.3, 84.9, 88.8, 84.6, 94.4, 84.1, 93.2, 90.4, 94.1, 78.3, 86.4, 83.6, 96.1, 83.7, 90.6, 89.1,
            97.8, 89.6, 85.1, 85.4, 98.0, 82.9, 91.4, 87.3, 93.1, 90.3, 84.0, 89.7, 85.4, 87.3, 88.2, 84.1, 86.4, 93.1,
            93.7, 87.6, 86.6, 86.4, 86.1, 90.1, 87.6, 94.6, 87.7, 85.1, 91.7, 84.5, 95.1, 95.2, 94.1, 96.3, 90.6, 89.6,
            87.5, 90.0, 86.1, 92.1, 94.7, 89.4, 90.0, 84.2, 96.4, 92.4, 94.3, 91.1, 88.6, 90.1, 85.1, 87.3, 93.2, 88.2,
            92.4, 84.1, 94.3, 90.5, 86.6, 86.7, 86.4, 90.6, 82.6, 97.3, 95.6, 91.2, 83.0, 85.0, 89.1, 83.1, 96.8, 88.3]
    print(describe(data))
    print(quantile(data, 0.3))
    print(cut(data, 2))
    print(statistics(data, 89, 2, 0.05))
