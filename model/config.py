import copy
import datetime
import hashlib
import shutil
from itertools import chain

import yaml
from yaml import Loader

from loader import *


@dataclass
class Default:
    """
    True means enabled, False means disabled
    """
    dataset: bool = True
    column: bool = False


@dataclass
class DataConfigLayout:
    root: Path
    configured: str = 'configured'
    ignored: str = 'ignored'
    backup: str = 'backup'

    def __post_init__(self):
        self.configured: Path = self.root.joinpath(self.configured)
        self.ignored: Path = self.root.joinpath(self.ignored)
        self.backup: Path = self.root.joinpath(self.backup)


def make_layout(layout: DataConfigLayout):
    root: Path = layout.__dict__['root']
    root = root.absolute()
    for key, value in layout.__dict__.items():
        if key == 'root':
            continue
        directory = root.joinpath(value)
        if not directory.exists():
            directory.mkdir()


def make_timestamp() -> str:
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def iter_dir(directory: Path,
             if_file: Callable[[Path], None],
             if_dir: Callable[[Path], None],
             if_else: Callable[[Path], None]):
    for path in directory.iterdir():
        if path.is_file():
            if_file(path)
        elif path.is_dir():
            if_dir(path)
        else:
            if_else(path)


def latest_file(paths: Iterable[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_ctime)


def hash_file(filename):
    """Generate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filename, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def make_zipfile(output_path: Path, source_dir: Path, exclude_dirs: list[Path], ignore_system_files=True):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_LZMA) as zipf:
        for root, dirs, files in os.walk(source_dir):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if root_path.joinpath(d) not in exclude_dirs]
            for file in files:
                file_path = root_path.joinpath(file)
                if ignore_system_files:
                    if file_path.stem == '.DS_Store':
                        # '.DS_Store': custom attributes of its containing folder, such as the positions of icons
                        continue
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)


class DataConfig:
    def __init__(self, layout: DataConfigLayout, default: Default = Default()):
        self.layout: DataConfigLayout = layout
        self.default = default

        self.unconfigured: list[Path] = []
        self.configured: list[Path] = []
        self.ignored: list[Path] = []

        self.original_csmar_datas: list[CsmarData] = []
        self.derived_csmar_datas: list[CsmarData] = []

    @property
    def combined_configs(self) -> chain[Path]:
        return chain(self.unconfigured, self.configured, self.ignored)

    def auto_config(self, csmar_data_dir: PathLike | str):
        """
        Automatically make new config files, perform any necessary cleanup operations and finally load them
        :return:
        """
        self.make_backup()

        self.load_config()
        self.original_csmar_datas = load_csmar_data(csmar_data_dir)

        derived_datas_dict: dict[str, CsmarData] = {d.csmar_datasheet.data_name: d for d in self.derived_csmar_datas}
        to_make: list[CsmarData] = []
        for original_data in self.original_csmar_datas:
            name = original_data.csmar_datasheet.data_name
            if name in derived_datas_dict:
                del derived_datas_dict[name]
            else:
                to_make.append(original_data)

        self.make_config(to_make)

        for lonely_data in derived_datas_dict.values():
            os.remove(lonely_data.config_path)

        self.load_config()
        self.clean_config()

    def make_config(self, datas: list[CsmarData]):
        """
        make config files
        :param datas:
        :return:
        """
        suffix = '.yaml'
        for data in datas:
            serialized_datasheet = data.serialize()
            name = data.csmar_datasheet.data_name
            save_to = self.layout.root.joinpath(name + suffix)
            with open(save_to, 'w') as f:
                yaml.dump(serialized_datasheet, f, sort_keys=False)
            print(f"Saved '{name}' to '{save_to}'")

    def clean_config(self):
        """
        Put existing config files into correct directories
        :return:
        """
        for data in self.derived_csmar_datas:
            config_path = data.config_path
            if self._is_dataset_ignored(data):
                if config_path.parent != self.layout.ignored:
                    shutil.move(config_path, self.layout.ignored)
            elif self._is_dataset_configured(data):
                if config_path.parent != self.layout.configured:
                    shutil.move(config_path, self.layout.configured)
            else:
                if config_path.parent != self.layout.root:
                    shutil.move(config_path, self.layout.root)

    def _is_dataset_ignored(self, data: CsmarData) -> bool:
        b = data.csmar_datasheet.disabled
        if b is None:
            return not self.default.dataset  # because we are testing if it is 'ignored'
        else:
            return b

    @staticmethod
    def _is_dataset_configured(data: CsmarData) -> bool:
        if data.csmar_datasheet.disabled is not None:
            return True
        for column in data.csmar_datasheet.column_infos:
            if column.enabled is not None:
                return True
        return False

    def load_config(self):
        self.derived_csmar_datas = []
        self._find_config_files()
        for config in self.combined_configs:
            with open(config, mode='r') as f:
                y = yaml.load(f, Loader=Loader)
                csmar_data = CsmarData.deserialize(y)
                csmar_data.config_path = config
                self.derived_csmar_datas.append(csmar_data)
        print(f"Loaded {len(self.derived_csmar_datas)} config files:")
        print('\n'.join([d.csmar_datasheet.data_name for d in self.derived_csmar_datas]))

    def _find_config_files(self):
        self.configured = []
        self.configured = []
        self.ignored = []

        def raise_error(p):
            raise RuntimeError(f"Config dir damaged: do not expect {p}")

        def if_file(append_to: list) -> Callable[[Path], None]:
            def f(p: Path):
                if p.is_file():
                    if p.suffix == '.yaml':
                        append_to.append(p)
                    elif p.stem == '.DS_Store':
                        return
                    else:
                        raise_error(p)

            return f

        def if_dir(p: Path):
            legitimate_dirs = self.layout.__dict__.values()
            if p not in legitimate_dirs:
                raise_error(p)

        iter_dir(self.layout.root, if_file(self.unconfigured), if_dir, raise_error)

        iter_dir(self.layout.configured, if_file(self.configured), raise_error, raise_error)

        iter_dir(self.layout.ignored, if_file(self.ignored), raise_error, raise_error)

        dic: dict = copy.deepcopy(self.layout.__dict__)
        del dic['root']
        for path in dic.values():
            path: Path = path
            if not (path.is_dir() and path.exists()):
                raise RuntimeError(f"Config dir damaged: do not found {path}")

    def make_backup(self):
        # last_backup = latest_file(self.layout.backup.iterdir())
        # old_hash = hash_file(last_backup)

        output_path = self.layout.backup.joinpath(make_timestamp() + '.zip')
        exclude = [self.layout.backup]
        make_zipfile(output_path, self.layout.root, exclude)
        #
        # new_hash = hash_file(output_path)
        # if new_hash == old_hash:
        #     os.remove(output_path)


if __name__ == '__main__':
    # make_layout(DataConfigLayout(Path('./config/data')))
    # csmar_datas = load_csmar_data(r'/Users/a/playground/freestyle/')
    DataConfig(DataConfigLayout(Path('./config/data'))).auto_config(r'/Users/a/playground/freestyle/')
