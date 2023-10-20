import numpy as np
from scipy import stats


def stats_median(vec: np.ndarray, m0: float, kind: str, level: float) -> tuple[float, str, str]:
    sign = vec - m0
    sign = sign[sign != 0]
    sign = sign > 0
    n = np.size(sign)
    p_value = stats.binom.pmf(sign.sum(), n, 0.5)

    if kind == '<':
        p_value = p_value
    elif kind == '>':
        p_value = 1 - p_value
    else:
        p_value = 2 * p_value if sign.sum() < n / 2 else 2 * (1 - p_value)

    return p_value, f'm {kind} m0', 'Bac bo' if p_value < level else 'Khong bac bo'


if __name__ == '__main__':
    data = np.loadtxt('data.txt', delimiter=' ')
    print(stats_median(data, 2000, '=', 0.05))