import os
import re
import zipfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable

import pandas as pd
from typing_extensions import Iterable


@dataclass
class CsmarDirectory:
    csmar_dir: Path
    data: Path
    datasheet: Path
    copy_right: Path


@dataclass
class CsmarColumnInfo:
    column_name: str
    full_name: str
    explanation: str


class CsmarData:
    def __init__(self, csmar_directory: CsmarDirectory):
        self.csmar_directory = csmar_directory
        self.csmar_datasheet = CsmarDatasheet(self.csmar_directory.datasheet)


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
        self.data_name = data_name[0].strip()

    def _load_column_infos(self):
        with open(self.datasheet_path, 'r') as datasheet:
            for line in datasheet:
                # let's consider this: 'Gdpq0102 [Gdp - Primary Industry] - Calculated at current prices'
                # I omit /n because the last column may not have that.

                explanation = re.findall(r'] - (.*)$', line)
                if len(explanation) != 1:
                    if len(explanation) == 0:
                        self.column_infos[-1].explanation += line
                        continue
                        # Datatype [Data Type] - 1=Industry value added (calculated at current price, unit: CNY100 million);
                        # 2=Growth of industry value added
                    raise RuntimeError(f'There should be exactly one after " - ", found {len(explanation)} in {line} '
                                       f'at {self.datasheet_path}')
                explanation = explanation[0]

                full_name = re.findall(r'\[(.*?)]', line)
                # ? here is to make the match not greedy, so nested [] are not allowed
                if len(full_name) != 1:
                    raise RuntimeError(f"There should be exactly one pair of [], found {len(full_name)} in {line} at "
                                       f"{self.datasheet_path}")
                full_name = full_name[0]
                full_name = full_name.strip()

                column_name = re.findall(r'(.*?)\[.*?]', line)
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
        if len(list(directory.iterdir())) == 0:
            pass
        elif examine_csmar_dir(directory):
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


def examine_csmar_dir(directory: PathLike | str) -> CsmarDirectory | None:
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

    csmar_dir = directory.absolute()
    copyright_path = None
    datasheet_path = None
    data_path = []

    files = list(directory.iterdir())
    for file in files:
        if file.name == copyright_file:
            copyright_count += 1
            copyright_path = file.absolute()
        elif file.suffix == '.txt':
            txt_count += 1
            datasheet_path = file.absolute()
        elif file.suffix == '.csv':
            csv_count += 1
            data_path.append(file.absolute())

    if csv_count > 1:
        names = [p.name for p in data_path]
        main_name = None
        count = 0
        for name in names:
            if not re.match(r'.*\d+\.csv$', name):
                if main_name is None:
                    main_name = name
                else:
                    raise RuntimeError('Duplicate main name found')
            else:
                count += 1

        if main_name:
            count += 1
        else:
            raise RuntimeError('Cannot find main name')
        if count != len(data_path):
            raise RuntimeError('Unknown situation')

        save_to = data_path[0].parent.joinpath(main_name)
        comb = pd.concat(map(pd.read_csv, data_path))  # todo performance optimization

        for p in data_path:
            os.remove(p)

        comb.to_csv(save_to, index=False)

        data_path = [save_to]
        csv_count = 1

    if copyright_count == csv_count == txt_count == 1:
        return CsmarDirectory(csmar_dir, data_path[0], datasheet_path, copyright_path)
    else:
        return None


def load_csmar_data(directory: PathLike | str) -> list[CsmarData]:
    zip_files = filter_zip_files(directory)
    data_dirs: Iterable[CsmarDirectory] = map(examine_csmar_dir, unzip_all(zip_files))
    csmar_data: list[CsmarData] = list(map(CsmarData, data_dirs))
    print(f"Found {len(csmar_data)} csmar data directories")
    return csmar_data


if __name__ == '__main__':
    # CsmarDatasheet(
    #     r'/Users/a/playground/freestyle/China Economic Research Series/Population Aging/Population/Population by Country_Region175929675/PAG_CounRegPopY[DES][csv].txt')
    load_csmar_data(r'/Users/a/playground/freestyle/')
    # filter_zip_files(r'/Users/a/playground/freestyle/')
    # examine_csmar_dir(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220')
    # unzip(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220.zip')
