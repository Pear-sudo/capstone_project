import json
import os
import re
import sys
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, TypeVar, Optional

import pandas as pd
from typing_extensions import Iterable

"""
This module is responsible for loading the csv data.
"""


def print_if(condition: bool, message: str) -> None:
    if condition:
        print(message)


def head(filename: Path, n=10) -> list[str]:
    with open(filename, 'r', encoding='utf-8') as file:
        lines = []
        for _ in range(n):
            line = file.readline().strip()
            if not line:
                break
            lines.append(line)
        return lines


def tail(filename: Path, n=10) -> list[str]:
    with open(filename, 'rb') as file:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        buffer_size = 1024
        data = ''
        lines_found = []

        while file.tell() > 0 and len(lines_found) < n:
            cursor = min(buffer_size, file.tell())
            file.seek(-cursor, 1)  # move up a bit
            data = file.read(cursor).decode('utf-8') + data
            file.seek(-cursor, 1)  # as read operation moved the position down
            lines_found = data.splitlines()

        return lines_found[-n:]  # it is possible that one bunch of data contains more than one lines


def remove_quotes(text: str) -> str:
    return text.replace('"', '').replace('\uFEFF', '')


def remove_all_quotes(text: list[str]) -> list[str]:
    return [remove_quotes(t) for t in text]


T = TypeVar('T', bound='Serializable')


class Serializable(ABC):
    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def reconcile(self) -> T:
        pass

    @staticmethod
    @abstractmethod
    def struct() -> dict[str, None]:
        pass

    @staticmethod
    @abstractmethod
    def deserialize(data: dict) -> T:
        pass


class Notation:
    ENABLED = ['y', '1']
    DISABLED = ['n', '0']

    @classmethod
    def deserialize(cls, data: str) -> Optional[bool]:
        if data == '':
            return None
        elif data in cls.ENABLED:
            return True
        elif data in cls.DISABLED:
            return False
        else:
            raise ValueError(f'"{data}" is not a valid notation')

    @classmethod
    def serialize(cls, data: Optional[bool]) -> str:
        if data is None:
            return ''
        elif data is True:
            return 'y'
        elif data is False:
            return 'n'
        else:
            raise ValueError(f'"{data}" is not a valid notation')


@dataclass
class CsmarDirectory:
    csmar_dir: Path
    data: Path
    datasheet: Path
    copy_right: Path


@dataclass
class CsmarColumnInfo(Serializable):
    column_name: str
    full_name: str
    explanation: str
    enabled: bool = None
    filter: str = ''
    instruction: str = ''

    @staticmethod
    def struct() -> dict[str, None]:
        return {}

    def serialize(self) -> dict:
        return {
            'column_name': self.column_name,
            'enabled': Notation.serialize(self.enabled),
            'full_name': self.full_name,
            'explanation': self.explanation,
            'filter': self.filter,
            'instruction': self.instruction
        }

    def reconcile(self) -> T:
        pass

    @staticmethod
    def deserialize(data: dict) -> 'CsmarColumnInfo':
        column_name: str = data['column_name']
        full_name: str = data['full_name']
        explanation: str = data['explanation']
        enabled: bool = Notation.deserialize(data['enabled'])
        column_info = CsmarColumnInfo(column_name, full_name, explanation, enabled)
        column_info.filter = data['filter']
        column_info.instruction = data['instruction']
        return column_info


class CsmarData(Serializable):

    def __init__(self, csmar_directory: CsmarDirectory = None, manual: bool = False):
        if not manual:
            self.csmar_directory = csmar_directory
            self.csmar_datasheet = CsmarDatasheet(self.csmar_directory.datasheet)

            self.config_path: Optional[Path] = None

            self.is_structure_updated: bool = False

    @staticmethod
    def struct() -> dict[str, None]:
        return {
            **CsmarDatasheet.struct(),
            'head': None,
            'tail': None
        }

    def serialize(self) -> dict:
        return {
            **self.csmar_datasheet.serialize(),
            'head': remove_all_quotes(head(self.csmar_directory.data)),
            'tail': remove_all_quotes(tail(self.csmar_directory.data))
        }

    def core_serialize(self) -> dict:
        """
        Do not read additional data from the disk.
        :return:
        """
        return {
            **self.csmar_datasheet.serialize(),
        }

    def reconcile(self) -> T:
        pass

    @staticmethod
    def deserialize(data: dict) -> 'CsmarData':
        data = CsmarData.remove_irrelevant_data(data)

        csmar_directory_path = data['path']
        csmar_directory = examine_csmar_dir(csmar_directory_path)
        csmar_data = CsmarData(manual=True)
        csmar_data.csmar_directory = csmar_directory

        csmar_datasheet = CsmarDatasheet.deserialize(data)

        csmar_data.csmar_datasheet = csmar_datasheet

        new_data = csmar_data.core_serialize()
        csmar_data.is_structure_updated = json.dumps(new_data) != json.dumps(data)

        return csmar_data

    @staticmethod
    def remove_irrelevant_data(data: dict) -> dict:
        del data['head']
        del data['tail']
        return data


class CsmarDatasheet(Serializable):

    def __init__(self, datasheet_path: PathLike | str = None, manual: bool = False):
        if not manual:
            self.datasheet_path = Path(datasheet_path)
            if not self.datasheet_path.is_file() and self.datasheet_path.suffix == ".txt":
                raise ValueError(f"Datasheet file {datasheet_path} is not valid")
            self.data_name: str = ''
            self.column_infos: list[CsmarColumnInfo] = []
            self.disabled: Optional[bool] = None

            self._load_data_name()
            self._load_column_infos()

    @staticmethod
    def struct() -> dict[str, None]:
        return {}

    def serialize(self) -> dict:
        columns: list = [column.serialize() for column in self.column_infos]
        return {
            'name': self.data_name,
            'path': str(self.datasheet_path.parent.absolute()),
            'disabled': Notation.serialize(self.disabled),
            'columns': columns
        }

    def reconcile(self) -> T:
        pass

    @staticmethod
    def deserialize(data: dict) -> 'CsmarDatasheet':
        csmar_datasheet = CsmarDatasheet(manual=True)
        csmar_datasheet.datasheet_path = examine_csmar_dir(data['path']).datasheet
        csmar_datasheet.data_name = data['name']
        csmar_datasheet.disabled = Notation.deserialize(data['disabled'])

        columns_data = data['columns']
        columns = [CsmarColumnInfo.deserialize(d) for d in columns_data]

        csmar_datasheet.column_infos = columns

        return csmar_datasheet

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


def unzip_all(zip_files: list[Path], silence: bool = False) -> list[Path]:
    non_zip_files = [str(file) for file in zip_files if not is_zipfile(file)]
    if non_zip_files:
        raise ValueError("Those files are not zip:\n {}".format('\n'.join(non_zip_files)))

    unzipped_to: list[Path] = []
    for zip_file in zip_files:
        path = unzip(zip_file, silence=silence)
        unzipped_to.append(path)

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


def unzip(file: str | PathLike, silence: bool = False) -> Path:
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
            # if already unzipped
            print_if(not silence, f'Skipping "{zip_path}" as it has been unzipped')
            return directory
        else:
            raise RuntimeError(f'Target directory "{directory}" exists but its content is dubious')
    elif not directory.exists():
        directory.mkdir(parents=False, exist_ok=False)
    else:
        raise RuntimeError('Unexpected situation')

    print_if(not silence, 'Unzipping "{}"'.format(zip_path.name))
    with zipfile.ZipFile(zip_path, mode='r') as zip_ref:
        zip_ref.extractall(directory)
    print_if(not silence, 'Unzipped "{}"'.format(zip_path.name))

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
    csmar_datas: list[CsmarData] = list(map(CsmarData, data_dirs))
    csmar_datas = examine_csmar_datas(csmar_datas, auto_correct=True)
    print(f"Found {len(csmar_datas)} csmar data directories")
    return csmar_datas


def examine_zip_files(files: list[Path]):
    name_set = set(file.stem for file in files)
    if len(name_set) < len(files):
        raise RuntimeError(f'Found {len(files) - len(name_set)} duplicate zip files')


def examine_csmar_datas(datas: list[CsmarData], auto_correct: bool = False) -> list[CsmarData]:
    data_name_set = set(data.csmar_datasheet.data_name for data in datas)
    discrepancy = len(datas) - len(data_name_set)
    if discrepancy != 0:
        count = Counter(data.csmar_datasheet.data_name for data in datas)
        erratic_data_names = {k for k, v in count.items() if v > 1}
        erratic_data: dict[str, list[CsmarData]] = {}
        for data in datas:
            name = data.csmar_datasheet.data_name
            if name in erratic_data_names:
                erratic_data.setdefault(name, []).append(data)

        for k, v in erratic_data.items():
            print(f"The following data shares a same name ('{k}'):\n")
            for data in v:
                print(f'\t{data.csmar_directory.csmar_dir}')
            print()
        sys.stdout.flush()
        if not auto_correct:
            # TODO handle it properly rather than simply drop the data
            raise RuntimeError(f'Csmar data names do not match, {len(data_name_set)} unique, {len(datas)} in list')
        else:
            return [data for data in datas if data.csmar_datasheet.data_name not in erratic_data_names]
    return datas


if __name__ == '__main__':
    # CsmarDatasheet(
    #     r'/Users/a/playground/freestyle/China Economic Research Series/Population Aging/Population/Population by Country_Region175929675/PAG_CounRegPopY[DES][csv].txt')
    load_csmar_data(r'/Users/a/playground/freestyle/')
    # filter_zip_files(r'/Users/a/playground/freestyle/')
    # examine_csmar_dir(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220')
    # unzip(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220.zip')
