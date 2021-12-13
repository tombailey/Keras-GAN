import sys

from os import listdir
from os.path import join

from PIL import Image


def init():
    folder = sys.argv[1]
    for file in listdir(folder):
        trim_skin(join(folder, file))


def trim_skin(file):
    image = Image.open(file).convert('RGBA')
    trim_areas = [
        {
            'x': {
                'start': 0,
                'end': 8
            },
            'y': {
                'start': 0,
                'end': 8
            }
        },
        {
            'x': {
                'start': 24,
                'end': 40
            },
            'y': {
                'start': 0,
                'end': 8
            }
        },
        {
            'x': {
                'start': 56,
                'end': 64
            },
            'y': {
                'start': 0,
                'end': 8
            }
        },
        {
            'x': {
                'start': 0,
                'end': 4
            },
            'y': {
                'start': 16,
                'end': 20
            }
        },
        {
            'x': {
                'start': 12,
                'end': 20
            },
            'y': {
                'start': 16,
                'end': 20
            }
        },
        {
            'x': {
                'start': 36,
                'end': 44
            },
            'y': {
                'start': 16,
                'end': 20
            }
        },
        {
            'x': {
                'start': 52,
                'end': 64
            },
            'y': {
                'start': 16,
                'end': 20
            }
        },
        {
            'x': {
                'start': 56,
                'end': 64
            },
            'y': {
                'start': 16,
                'end': 40
            }
        },
        {
            'x': {
                'start': 0,
                'end': 4
            },
            'y': {
                'start': 32,
                'end': 36
            }
        },
        {
            'x': {
                'start': 12,
                'end': 20
            },
            'y': {
                'start': 32,
                'end': 36
            }
        },
        {
            'x': {
                'start': 36,
                'end': 44
            },
            'y': {
                'start': 32,
                'end': 36
            }
        },
        {
            'x': {
                'start': 52,
                'end': 64
            },
            'y': {
                'start': 32,
                'end': 36
            }
        },
        {
            'x': {
                'start': 0,
                'end': 4
            },
            'y': {
                'start': 48,
                'end': 52
            }
        },
        {
            'x': {
                'start': 12,
                'end': 20
            },
            'y': {
                'start': 48,
                'end': 52
            }
        },
        {
            'x': {
                'start': 28,
                'end': 36
            },
            'y': {
                'start': 48,
                'end': 52
            }
        },
        {
            'x': {
                'start': 44,
                'end': 52
            },
            'y': {
                'start': 48,
                'end': 52
            }
        },
        {
            'x': {
                'start': 60,
                'end': 64
            },
            'y': {
                'start': 48,
                'end': 52
            }
        },
    ]

    for trim_area in trim_areas:
        for x in range(trim_area['x']['start'], trim_area['x']['end']):
            for y in range(trim_area['y']['start'], trim_area['y']['end']):
                image.putpixel((x, y), (0, 0, 0, 0))
    image.save(file, quality=100)


if __name__ == "__main__":
    init()
