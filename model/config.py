import copy
import datetime
import shutil

import yaml

from loader import *


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


class DataConfig:
    def __init__(self, layout: DataConfigLayout):
        self.layout: DataConfigLayout = layout

        self.unconfigured: list[Path] = []
        self.configured: list[Path] = []
        self.ignored: list[Path] = []

    def make_config(self, datas: list[CsmarData]):
        suffix = '.yaml'
        for data in datas:
            serialized_datasheet = data.serialize()
            name = data.csmar_datasheet.data_name
            save_to = self.layout.root.joinpath(name + suffix)
            with open(save_to, 'w') as f:
                yaml.dump(serialized_datasheet, f, sort_keys=False)
            print(f"Saved '{name}' to '{save_to}'")

    def clean_config(self):
        pass

    def load_config(self):
        pass

    def _find_config_files(self):
        def raise_error(p):
            raise RuntimeError(f"Config dir damaged: do not expect {p}")

        def if_file(append_to: list) -> Callable[[Path], None]:
            def f(p: Path):
                if p.is_file():
                    if p.suffix == '.yaml':
                        append_to.append(p)
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
        file_path = self.layout.backup
        shutil.make_archive(dry_run=True, base_name=make_timestamp(), root_dir=self.layout.backup, )


if __name__ == '__main__':
    # make_layout(DataConfigLayout(Path('./config/data')))
    # csmar_datas = load_csmar_data(r'/Users/a/playground/freestyle/')
    DataConfig(DataConfigLayout(Path('./config/data'))).make_backup()
