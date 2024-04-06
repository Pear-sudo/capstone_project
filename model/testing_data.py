from pathlib import Path

import numpy as np
import pandas as pd


def generate_testing_data() -> pd.DataFrame:
    np.random.seed(0)
    size = 10_000

    a = np.random.randint(1, 10, size)
    b = np.random.randint(1, 10, size)
    c = np.random.randint(1, 10, size)

    # x = np.sin(a) + np.sin(b) + np.sin(c) - 2 * b
    # y = np.power(a, 2) + np.power(b, 2) + np.power(c, 3)
    # z = a * b + np.log(c)

    x = 2 * a
    y = a * b * c
    z = np.power(b, 2) - x * c - c

    x = np.roll(x, 1)
    y = np.roll(y, 1)
    z = np.roll(z, 1)

    d = pd.DataFrame({
        'a': a,
        'b': b,
        'c': c,
        'x': x,
        'y': y,
        'z': z
    })

    return d[:-1]


if __name__ == '__main__':
    out_dir = Path('../out/test')
    if not out_dir.exists():
        out_dir.mkdir()

    df = generate_testing_data()
    df.to_csv(out_dir.joinpath('testing_data.csv'), index=False)
