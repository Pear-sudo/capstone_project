import pandas as pd
from pathlib import Path

from model.loader import filter_files


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


def get_dropped(path: Path):
    r2_paths = filter_files(path, lambda s: Path(s).suffix == '.txt')
    names = []
    r2s = []
    for r2_path in r2_paths:
        name = r2_path.parent.name
        if name == 'result':
            name = 'complete'
        with open(r2_path, 'r') as f:
            s = f.readline().strip()
            r2 = float(s)
            r2s.append(r2)
        names.append(name)
    df = pd.DataFrame(zip(names, r2s), columns=['name', 'r2'])
    save_path = path.joinpath('dropped_r2.xlsx')
    df.to_excel(save_path, index=False)


if __name__ == '__main__':
    real = Path('/Users/a/PycharmProjects/capstone/capstone project/model/checkpoints_real/result')
    random = Path('/Users/a/PycharmProjects/capstone/capstone project/model/checkpoints/result')
    dropped = Path('/Users/a/PycharmProjects/capstone/capstone project/model/incomplete/result')
    dropped_random = Path('/Users/a/PycharmProjects/capstone/capstone project/model/incomplete_random/result')
    get_dropped(dropped_random)
    # get_r2(random)
