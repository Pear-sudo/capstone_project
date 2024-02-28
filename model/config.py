from loader import *


@dataclass
class DataConfigLayout:
    root: Path
    configured: str = 'configured'
    ignored: str = 'ignored'


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
    pass
