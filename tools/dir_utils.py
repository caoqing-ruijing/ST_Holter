import os
from natsort import natsorted
from glob import glob
import shutil


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir_without_del(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdir_with_del(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x