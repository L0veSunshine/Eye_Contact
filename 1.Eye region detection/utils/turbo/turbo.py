import os
import platform
from utils.turbo.TurboJPEG import TurboJPEG


def serach(path):
    if platform.system() == "Windows":
        target_name = 'turbojpeg.dll'
    elif platform.system() == "Linux":
        target_name = 'libturbojpeg.so'
    else:
        raise Exception("system was not supported.")

    res = ''

    def inner(subpath):
        nonlocal res
        files = os.scandir(subpath)
        for f in files:
            if f.is_file() and f.name == target_name:
                res = f.path
                break
            elif f.is_dir():
                inner(f.path)

    inner(path)
    return res


reader = TurboJPEG(serach(os.getcwd()))
