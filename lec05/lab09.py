import numpy as np
import math


def stats_median(vec: np.ndarray, m0: float, kind: str, level: float) -> tuple[float, str, str]:
    sign = vec - m0
    sign = sign[sign != 0]
    sign = sign > 0
    n = np.size(sign)
    r = sign.sum()
    p_value = sum(math.comb(n, x) * (0.5 ** x) * ((1 - 0.5) ** (n - x)) for x in range(r, n + 1))

    if kind == '<':
        p_value = 1 - p_value
    elif kind == '>':
        p_value = p_value
    else:
        p_value = 2 * (1 - p_value) if sign.sum() < n / 2 else 2 * p_value

    return p_value, f'm {kind} m0', 'Reject' if p_value < level else 'Not reject'


if __name__ == '__main__':
    data = np.loadtxt('data.txt', delimiter=' ')
    print(stats_median(data, 2000, '=', 0.05))
