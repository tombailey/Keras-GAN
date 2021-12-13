import hashlib

import sys

from os import listdir, remove
from os.path import join


def init():
    folder = sys.argv[1]
    hash_set = set()
    for file in listdir(folder):
        file_hash = sha256sum(join(folder, file))
        if file_hash in hash_set:
            print(f'{file} is a duplicate')
            remove(join(folder, file))
        else:
            hash_set.add(file_hash)


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


if __name__ == "__main__":
    init()
