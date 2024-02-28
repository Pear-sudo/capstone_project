import os
import re
import zipfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable


def config_csmar_data(directory: PathLike | str):
    zip_files = filter_zip_files(directory)
    data_dirs = unzip_all(zip_files)
    print(f"Found {len(data_dirs)} data directories")


@dataclass
class CsmarColumnInfo:
    column_name: str
    full_name: str
    explanation: str


class CsmarDatasheet:
    def __init__(self, datasheet_path: PathLike | str):
        self.datasheet_path = Path(datasheet_path)
        if not self.datasheet_path.is_file() and self.datasheet_path.suffix == ".txt":
            raise ValueError(f"Datasheet file {datasheet_path} is not valid")
        self.data_name: str = ''
        self.column_infos: list[CsmarColumnInfo] = []
        self._load_data_name()
        self._load_column_infos()

    def _load_data_name(self):
        dir_path = self.datasheet_path.parent
        if not dir_path.is_dir():
            raise ValueError(f"Datasheet {self.datasheet_path} is not residing in a standard csmar directory")
        dir_name = dir_path.name
        data_name = re.findall(r'(.*)\d{9}', dir_name)
        if len(data_name) != 1:
            raise ValueError(f"{dir_path} is not a valid csmar directory")
        self.data_name = data_name[0]

    def _load_column_infos(self):
        with open(self.datasheet_path, 'r') as datasheet:
            for line in datasheet:
                splits = line.split(' - ')
                if len(splits) != 2:
                    raise RuntimeError(f"Unexpected line {line} in {self.datasheet_path}")

                combo = splits[0]
                explanation = splits[1]
                if explanation == "\n":
                    explanation = ""

                full_name = re.findall(r'\[(.*?)]', combo)
                # ? here is to make the match not greedy, so nested [] are not allowed
                if len(full_name) != 1:
                    raise RuntimeError(f"There should be exactly one pair of [], found {len(full_name)} in {line} at "
                                       f"{self.datasheet_path}")
                full_name = full_name[0]
                full_name = full_name.strip()

                column_name = re.findall(r'(.*?)\[.*?]', combo)
                if len(column_name) != 1:
                    raise RuntimeError(f"There should be exactly one name before [], found {len(full_name)} in {line} "
                                       f"at {self.datasheet_path}")
                column_name = column_name[0]
                column_name = column_name.strip()

                column_info = CsmarColumnInfo(column_name, full_name, explanation)
                self.column_infos.append(column_info)


def unzip_all(zip_files: list[Path]) -> list[Path]:
    non_zip_files = [str(file) for file in zip_files if not is_zipfile(file)]
    if non_zip_files:
        raise ValueError("Those files are not zip:\n {}".format('\n'.join(non_zip_files)))

    unzipped_to: list[Path] = []
    for zip_file in zip_files:
        print('Unzipping "{}"'.format(zip_file.name))
        path = unzip(zip_file)
        unzipped_to.append(path)
        print('Unzipped "{}"'.format(zip_file.name))

    return unzipped_to


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
    CsmarDatasheet(
        r'/Users/a/playground/freestyle/China Economic Research Series/Population Aging/Population/Population by Country_Region175929675/PAG_CounRegPopY[DES][csv].txt')
    # config_csmar_data(r'/Users/a/playground/freestyle/')
    # filter_zip_files(r'/Users/a/playground/freestyle/')
    # examine_csmar_dir(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220')
    # unzip(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220.zip')
