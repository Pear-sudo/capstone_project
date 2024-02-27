import zipfile
from os import PathLike
from pathlib import Path


def unzip(file: str) -> Path:
    zip_path = Path(file)
    if not zip_path.is_file():
        raise FileNotFoundError()
    if not zip_path.suffix == '.zip':
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
    # examine_csmar_dir(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220')
    # unzip(r'/Users/a/playground/freestyle/China Economic Research Series/Macroeconomic/Gdp/Quarterly Gross Domestic Product181521220.zip')
