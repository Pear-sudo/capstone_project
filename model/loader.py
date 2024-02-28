import os
import zipfile
from os import PathLike
from pathlib import Path
from typing import Callable


def unzip_all(zip_files: list[Path]) -> bool:
    non_zip_files = [str(file) for file in zip_files if not is_zipfile(file)]
    if non_zip_files:
        raise ValueError("Those files are not zip:\n {}".format('\n'.join(non_zip_files)))

    for zip_file in zip_files:
        print('Unzipping {}'.format(zip_file.name))
        unzip(zip_file)
        print('Unzipped {}'.format(zip_file.name))

    return False


def filter_zip_files(directory: PathLike | str) -> list[Path]:
    return filter_files(directory, lambda file: file.endswith('.zip'))


def filter_files(directory: PathLike | str, examiner: Callable[[str], bool]) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'{directory} is not a directory')
    target_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # I choose to pass str to examiner for performance considerations
            # after all, most of these files will be discarded.
            if examiner(file):
                target_paths.append(Path(os.path.join(root, file).__str__()))
    return target_paths


def unzip(file: str | PathLike) -> Path:
    zip_path = Path(file)
    if not zip_path.is_file():
        raise FileNotFoundError()
    if not is_zipfile(zip_path):
        raise ValueError(f'File {file} is not a zip file')

    parent = zip_path.parent
    directory = zip_path.stem
    directory = parent.joinpath(directory)
    if directory.is_file():
        raise RuntimeError(f'Target directory {directory} is a file')
    elif directory.is_dir():
        if examine_csmar_dir(directory):
            return directory
        else:
            raise RuntimeError(f'Target directory {directory} exists but its content is dubious')
    elif not directory.exists():
        directory.mkdir(parents=False, exist_ok=False)
    else:
        raise RuntimeError('Unexpected situation')

    with zipfile.ZipFile(zip_path, mode='r') as zip_ref:
        zip_ref.extractall(directory)

    return directory


def is_zipfile(file: Path) -> bool:
    return file.suffix == '.zip'


def examine_csmar_dir(directory: PathLike | str) -> bool:
    """
    A typical csmar directory (with csv data) contains three files, for example:
    CME_Qgdp.csv
    CME_Qgdp[DES][csv].txt
    Copyright notice.pdf
    :param directory:
    :return:
    """
    if type(directory) is str:
        directory = Path(directory)

    if not directory.is_dir():
        raise ValueError(f'directory {directory} is not a directory')

    copyright_file = 'Copyright notice.pdf'
    copyright_count = 0
    csv_count = 0
    txt_count = 0

    files = list(directory.iterdir())
    if len(files) != 3:
        return False
        # raise ValueError(f'There should be three files in {directory}, but got {len(files)}')
    for file in files:
        if file.name == copyright_file:
            copyright_count += 1
        elif file.suffix == '.txt':
            txt_count += 1
        elif file.suffix == '.csv':
            csv_count += 1

    if copyright_count == csv_count == txt_count == 1:
        return True
    else:
        return False


if __name__ == '__main__':
    pass
    filter_zip_files(r'/Users/a/playground/freestyle/')
    # examine_csmar_dir(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220')
    # unzip(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220.zip')
