import yaml

from loader import *


@dataclass
class DataConfigLayout:
    root: Path
    configured: str = 'configured'
    ignored: str = 'ignored'
    backup: str = 'backup'


def make_layout(layout: DataConfigLayout):
    root: Path = layout.__dict__['root']
    root = root.absolute()
    for key, value in layout.__dict__.items():
        if key == 'root':
            continue
        directory = root.joinpath(value)
        if not directory.exists():
            directory.mkdir()


class DataConfig:
    def __init__(self, layout: DataConfigLayout):
        self.layout: DataConfigLayout = layout

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

    def read_config(self):
        pass


if __name__ == '__main__':
    # make_layout(DataConfigLayout(Path('./config/data')))
    csmar_datas = load_csmar_data(r'/Users/a/playground/freestyle/')
    DataConfig(DataConfigLayout(Path('./config/data'))).make_config(csmar_datas)
