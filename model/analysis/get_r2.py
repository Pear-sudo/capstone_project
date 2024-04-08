from pathlib import Path

import pandas as pd


def get_r2(path: Path):
    result_paths: list[Path] = []
    for p in path.iterdir():
        if p.is_dir():
            continue
        if p.suffix != '.txt':
            continue
        result_paths.append(p)

    result_dict: dict[str, list[tuple[str, float]]] = {}
    sorted_result_list: list[list] = [[]] * 4
    for p in result_paths:
        file_stem = p.stem
        width, network = file_stem.split('_')
        r2 = None
        with open(p, 'r') as f:
            s = f.readline().strip()
            r2 = float(s)
        result_dict.setdefault(width, []).append((network, r2))

    ordered_models = ['l',
                      'n3', 'n5', 'n7', 'nd7',
                      'nn3', 'nn5', 'nn7', 'ndnd7',
                      'nnn3', 'nnn5', 'nnn7', 'ndndnd7',
                      'cn3', 'cn5', 'cn7', 'cnd7', ]
    ordered_width = list(map(str, sorted(map(int, list(result_dict.keys())))))

    for width, model_r2s in result_dict.items():
        ordered_rs = [0] * len(ordered_models)
        for model, r2 in model_r2s:
            i = ordered_models.index(model)
            ordered_rs[i] = r2
        i = ordered_width.index(width)
        sorted_result_list[i] = ordered_rs

    sorted_result_list = list(zip(*sorted_result_list))

    df = pd.DataFrame(sorted_result_list, columns=ordered_width, index=ordered_models)
    df.to_excel(path.joinpath('r2.xlsx'))


if __name__ == '__main__':
    real = Path('/Users/a/PycharmProjects/capstone/capstone project/model/checkpoints_real/result')
    random = Path('/Users/a/PycharmProjects/capstone/capstone project/model/checkpoints/result')
    # get_r2(real)
    get_r2(random)
