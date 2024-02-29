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
        pass

    def clean_config(self):
        pass

    def read_config(self):
        pass


if __name__ == '__main__':
    make_layout(DataConfigLayout(Path('./config/data')))
