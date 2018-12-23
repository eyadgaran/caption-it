'''
Module for model training
'''

__author__ = 'Elisha Yadgaran'


# import plaidml.keras
# plaidml.keras.install_backend()
# import logging
# logging.getLogger("plaidml").setLevel(logging.CRITICAL)

from backend.database.initialization import SimpleMLDatabase
from backend.modeling.dataset import *
from backend.modeling.pipeline import *
from backend.modeling.transformer import *
from backend.modeling.constants import *

from simpleml.utils.training.create_persistable import RawDatasetCreator, DatasetPipelineCreator,\
    DatasetCreator, PipelineCreator, ModelCreator, MetricCreator
from simpleml.pipelines.validation_split_mixins import TRAIN_SPLIT, TEST_SPLIT
from simpleml.utils.scoring.load_persistable import PersistableLoader

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from itertools import product


def train():
    # Initialize session
    SimpleMLDatabase().initialize()

    # Main dataset
    text_raw_dataset_kwargs = {
        'project': 'captioner', 'name': 'coco_captions', 'strict': False,
        'registered_name': 'MSCocoStreamingCaptionsRawDataset',
        'label_columns': ['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6']
    }
    raw_dataset = RawDatasetCreator.retrieve_or_create(**text_raw_dataset_kwargs)


if __name__ == '__main__':
    train()
