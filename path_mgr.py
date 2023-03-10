import os

root_dir = os.path.dirname(__file__)


def segments_dir():
    return os.path.join(root_dir, "resources/segments_data")


def models_dir():
    return os.path.join(root_dir, "resources/models")
