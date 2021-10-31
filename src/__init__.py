import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


BACKGROUND_IDX = 0
LABEL_TO_TAG = {BACKGROUND_IDX: 'background', 1: 'cat', 2: 'dog'}
LABEL_TO_COLOR = {0: 'red', 1: 'yellow', 2: 'purple'}


MODEL_FILENAME_FMT = "{model_name}.epoch_{epoch}"
MODEL_STORAGE_DIR = os.path.join(os.getcwd(), "models")
REPORTS_STORAGE_DIR = os.path.join(os.getcwd(), "reports")
# MODEL_STORAGE_DIR = os.path.join(CACHE_BASE_DIR, "checkpoints")
# MODEL_TRAINING_PLOTS_DIR = os.path.join(CACHE_BASE_DIR, "plots")
BOX_FORMAT = Literal['xyxy', 'xywh', 'cxcywh']


def get_path_for_saving_model(model_name, epoch):
    path = os.path.join(
        MODEL_STORAGE_DIR,
        model_name,
        MODEL_FILENAME_FMT.format(model_name=model_name, epoch=epoch) + '.pt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_path_for_saving_plots(model_name, epoch):
    path = os.path.join(
        REPORTS_STORAGE_DIR,
        "figures",
        model_name,
        MODEL_FILENAME_FMT.format(model_name=model_name, epoch=epoch) + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
