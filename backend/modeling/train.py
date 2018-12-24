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

    # Encoder
    LATEST_TEXT_MODEL = PersistableLoader.load_model('text_model')
    image_dataset_pipeline_kwargs = {
        'project': 'captioner', 'name': 'image_pipeline', 'strict': False,
        'registered_name': 'UnsupervisedExplicitSplitDatasetPipeline',
        'transformers': [
              ('add_columns', AddDataframeColumns(columns=['coco_url', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6'])),
              ('drop_metadata', DropDataframeColumns(columns=['coco_url', 'y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6'], drop=False)),
              ('melt', PandasMelt(id_vars=['coco_url'], value_vars=['y_0', 'y_1', 'y_2', 'y_3', 'y_4', 'y_5', 'y_6'], value_name='y')),
              ('drop_stacked_id', DropDataframeColumns(columns=['variable'], drop=True)),
              ('drop_nulls', DropDataframeNulls()),
              ('tokenize', NLTKTweetTokenizer(column='y', strip_handles=True, preserve_case=False, reduce_len=True)),
              ('add_start_token', AddStartToken(column='y', token=START_TOKEN)),
              ('add_end_token', AddEndToken(column='y', token=END_TOKEN)),
              ('pad_sequences', PadSequence(column='y', token=PAD_TOKEN, pad_length=PAD_LENGTH)),
              ('convert_lists', ListsToArrays(column='y')),
              ('index_tokens', SimpleMLModelTransformer(name='text_model', version=LATEST_TEXT_MODEL.version, columns='y', output_column='y', drop=False)),
              ('duplicate_column', ColumnDuplicator(origin_column='y', destination_column='caption')),
              # Only add start to caption to keep y offset by one (predicting the next word)
              ('offset_y_values', OffsetValues(column='y', offset=1)),
              ('offset_caption_values', OffsetValues(column='caption', offset=-1, reverse=True)),
              ('rename_columns', RenameColumns(name_dict={'coco_url': 'image'})),
        ]
    }
    image_dataset_pipeline = DatasetPipelineCreator.retrieve_or_create(
        raw_dataset=raw_dataset, **image_dataset_pipeline_kwargs)

    image_dataset_kwargs = {
        'project': 'captioner', 'name': 'image_dataset', 'strict': False,
        'registered_name': 'MSCocoStreamingCaptionsEncodedDataset', 'label_columns': ['y']
    }
    image_dataset = DatasetCreator.retrieve_or_create(
        dataset_pipeline=image_dataset_pipeline, **image_dataset_kwargs)

    image_pipeline_kwargs = {
        'project': 'captioner', 'name': 'image_pipeline', 'strict': False,
        'registered_name': 'NoFitExplicitSplitProductionPipeline',
        'transformers': [
              ('load_images', ImageLoader(column='image')),
              ('crop', CropImageToSquares(column='image')),
              ('resize', ResizeImage(column='image', final_dims=(224, 224))),
              ('split_dataframe', SplitDataframe(columns=['image'])),
              ('dfs_to_matrices', TupleDataframesToMatrices()),
              ('preprocess_tuple', TupleKerasInceptionV3ImagePreprocessor(index=0)),
              ('encode', TupleInceptionV3Encoder(index=0)),
        ]
    }
    image_pipeline = PipelineCreator.retrieve_or_create(
        dataset=image_dataset, **image_pipeline_kwargs)

    # Decoder
    image_model_kwargs = {
        'project': 'captioner', 'name': 'image_model', 'strict': False,
        'registered_name': 'TrainingImageDecoder',
        'external_model_kwargs': {
            'vocabulary_size': len(LATEST_TEXT_MODEL.external_model.vocabulary_) + len(SPECIAL_TOKEN_MAP),
            'pad_length': PAD_LENGTH,
            'pad_index': SPECIAL_TOKEN_MAP[PAD_TOKEN]
        }
    }
    image_model = ModelCreator.retrieve_or_create(
        pipeline=image_pipeline, **image_model_kwargs)


if __name__ == '__main__':
    train()
