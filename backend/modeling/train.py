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

    # Text Processor
    text_dataset_pipeline_kwargs = {
        'project': 'captioner', 'name': 'text_pipeline', 'strict': False,
        'registered_name': 'UnsupervisedExplicitSplitDatasetPipeline',
        'transformers': [
              ('add_columns', AddDataframeColumns(columns=['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6'])),
              ('isolate_captions', DropDataframeColumns(columns=['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6'], drop=False)),
              ('stack_columns', StackDataframeColumns(name='captions'))
        ]
    }
    text_dataset_pipeline = DatasetPipelineCreator.retrieve_or_create(
        raw_dataset=raw_dataset, **text_dataset_pipeline_kwargs)

    text_dataset_kwargs = {
        'project': 'captioner', 'name': 'text_dataset', 'strict': False,
        'registered_name': 'MSCocoCaptionsDataset'
    }
    text_dataset = DatasetCreator.retrieve_or_create(
        dataset_pipeline=text_dataset_pipeline, **text_dataset_kwargs)

    text_pipeline_kwargs = {
        'project': 'captioner', 'name': 'text_pipeline', 'strict': False,
        'registered_name': 'BaseExplicitSplitProductionPipeline',
        'transformers': [
              ('squeeze_to_series', SqueezeTransformer()),
              ('tokenize', NLTKTweetTokenizer(strip_handles=True, preserve_case=False, reduce_len=True)),
              ('add_start_token', AddStartToken(token=START_TOKEN)),
              ('add_end_token', AddEndToken(token=END_TOKEN)),
              ('pad_sequences', PadSequence(token=PAD_TOKEN, pad_length=20)),
              ('convert_to_matrix', DataframeToMatrix()),
        ]
    }
    text_pipeline = PipelineCreator.retrieve_or_create(
        dataset=text_dataset, **text_pipeline_kwargs)

    text_model_kwargs = {
        'project': 'captioner', 'name': 'text_model', 'strict': False,
        'registered_name': 'TextProcessor',
        'external_model_kwargs': {'tokenizer': lambda token: token, 'min_df': 5,
                                  'decode_error': 'ignore', 'lowercase': False}
    }
    text_model = ModelCreator.retrieve_or_create(
        pipeline=text_pipeline, **text_model_kwargs)

if __name__ == '__main__':
    train()
