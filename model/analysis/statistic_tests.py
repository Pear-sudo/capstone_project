from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acovf


def dm_test(errors_model1: np.ndarray, errors_model2: np.ndarray) -> float:
    """
    Diebold-Mariano Test
    :return: a positive number means the SECOND (right hand side) model is better
    """
    d12 = errors_model1 ** 2 - errors_model2 ** 2
    mean_d12 = np.mean(d12)
    se_d12 = np.sqrt(acovf(d12, fft=True)[0] / len(d12))
    dm_stat = mean_d12 / se_d12
    return dm_stat


def get_errors(pair_path: Path) -> np.ndarray:
    df = pd.read_csv(pair_path)
    errors: pd.DataFrame = df['true_values'] - df['predicted_values']
    return errors.to_numpy()


def dm_one_width(width_path: Path) -> pd.DataFrame:
    ordered_models = ['l',
                      'n3', 'n5', 'n7', 'nd7',
                      'nn3', 'nn5', 'nn7', 'ndnd7',
                      'nnn3', 'nnn5', 'nnn7', 'ndndnd7',
                      'cn3', 'cn5', 'cn7', 'cnd7', ]
    paths: list[Path] = []
    for p in width_path.iterdir():
        if p.is_dir():
            continue
        if p.suffix != '.csv':
            continue
        paths.append(p)
    ordered_paths = sorted(paths, key=lambda p: ordered_models.index(p.stem))
    ordered_errors = [get_errors(p) for p in ordered_paths]

    errors_vertical = ordered_errors[:-1]
    models_vertical = ordered_models[:-1]

    errors_horizontal = ordered_errors[1:]
    models_horizontal = ordered_models[1:]

    dm_table = []

    for i, e_vertical in enumerate(errors_vertical):
        dms = []
        for e_horizontal in errors_horizontal[i:]:
            dm = dm_test(e_vertical, e_horizontal)
            dms.append(dm)
        target_length = len(errors_horizontal)
        delta = target_length - len(dms)
        for _ in range(delta):
            dms.insert(0, None)
        dm_table.append(dms)

    df = pd.DataFrame(dm_table, index=models_vertical, columns=models_horizontal)
    save_to = width_path.parent.joinpath(f'dm_{width_path.name}.xlsx')
    df.to_excel(save_to)
    return df


if __name__ == '__main__':
    p_7 = Path('/Users/a/PycharmProjects/capstone/capstone project/model/checkpoints_real/result/7')
    df = dm_one_width(p_7)
