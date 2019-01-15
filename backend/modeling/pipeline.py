'''
Module to define the pipeline(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.pipelines import BaseExplicitSplitPipeline
from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
from simpleml.utils.errors import PipelineError

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


''' Dataset Pipelines '''

class UnsupervisedExplicitSplitPipeline(BaseExplicitSplitPipeline):
    '''
    Converts supervised dataset into an usupervised one so transformers can manipulate labels
    '''
    def split_dataset(self):
        '''
        Method to split the dataframe into different sets. Assumes dataset
        explicitly delineates between train, validation, and test

        Additionally assumes dataframes returned with matching indices to execute
        `.join` on
        '''
        self._dataset_splits = {
            TRAIN_SPLIT: (self.dataset.get('X', TRAIN_SPLIT).join(self.dataset.get('y', TRAIN_SPLIT)), None),
            VALIDATION_SPLIT: (self.dataset.get('X', VALIDATION_SPLIT).join(self.dataset.get('y', VALIDATION_SPLIT)), None),
            TEST_SPLIT: (self.dataset.get('X', TEST_SPLIT).join(self.dataset.get('y', TEST_SPLIT)), None)
        }


''' Production Pipelines '''

class GeneratorPipeline(object):
    '''
    Extends Base Pipeline to support data generators for fitting
    '''
    def get_dataset_split(self, split=None, infinite_loop=False, batch_size=32, shuffle=True, **kwargs):
        '''
        Get specific dataset split
        '''
        if split is None:
            split = TRAIN_SPLIT

        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()

        # Data generators are formatted for keras models
        X, y = self._dataset_splits.get(split)

        dataset_size = X.shape[0]
        if isinstance(X, pd.DataFrame):
            indices = X.index.tolist()
        elif isinstance(X, np.ndarray):
            indices = np.arange(X.shape[0])
        else:
            raise NotImplementedError

        if dataset_size == 0:  # Return None
            return

        first_run = True
        current_index = 0
        while True:
            if current_index == 0 and shuffle and not first_run:
                np.random.shuffle(indices)

            batch = indices[current_index:min(current_index + batch_size, dataset_size)]

            if y is not None and (isinstance(y, (pd.DataFrame, pd.Series)) and not y.empty):
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    yield X.loc[batch], np.stack(y.loc[batch].squeeze().values)
                else:
                    yield X[batch], y[batch]
            else:
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    yield X.loc[batch], None
                else:
                    yield X[batch], None

            current_index += batch_size

            # Loop so that infinite batches can be generated
            if current_index >= dataset_size:
                if infinite_loop:
                    current_index = 0
                    first_run = False
                else:
                    break

    def transform(self, X, dataset_split=None, return_y=False, **kwargs):
        '''
        Pass through method to external pipeline

        :param X: dataframe/matrix to transform, if None, use internal dataset
        :param return_y: whether to return y with output - only used if X is None
            necessary for fitting a supervised model after
        '''
        if not self.state['fitted']:
            raise PipelineError('Must fit pipeline before transforming')

        if X is None:
            generator = self.get_dataset_split(dataset_split, **kwargs)
            for X_batch, y_batch in generator:
                output = self.external_pipeline.transform(X_batch, **kwargs)

                if return_y:
                    yield output, y_batch
                else:
                    yield output
        else:
            yield self.external_pipeline.transform(X, **kwargs)


class NoFitExplicitSplitPipeline(GeneratorPipeline, BaseExplicitSplitPipeline):
    def __init__(self, *args, **kwargs):
        super(NoFitExplicitSplitPipeline, self).__init__(*args, **kwargs)
        self.state['fitted'] = True

''' Utils '''
# Preprocess for training speed
# Map image files to preencoded ndarrays (dont do direct because mutliple indices map to the same image)
def encode_all_images(df, pipeline):
    '''
    Outputs to disk a directory with every file named {image_name}.npy
    '''
    # 1) Dedupe repeated images
    def dedupe(df):
        df.drop_duplicates(subset='image', keep='first', inplace=True)
        df['image_id'] = df.image.apply(lambda row: row.split('/')[-1])
        df.set_index('image_id', inplace=True)

        # Support resuming if operation crashes
        existing  = [i[:-4] for i in os.listdir('encoded_images') if i[-3:] == 'npy']
        df.drop(existing, axis=0, inplace=True)

        return df

    # 2) Batch process
    def batch_df(df, batch_size=100):
        df_size = len(df)
        indices = df.index.tolist()
        current_index = 0

        while current_index < df_size:
            next_batch_size = min(batch_size, df_size - current_index)
            next_indices = indices[current_index: current_index + next_batch_size]
            yield df.loc[next_indices]
            current_index += next_batch_size

    # 3) Run through pipeline
    def pipeline_batch(pipeline, batch):
        batch_indices = batch.index.tolist()
        transformed_batch = next(pipeline.transform(batch))[0]
        return dict(zip(batch_indices, transformed_batch))

    # 4) Save encoded images
    def save_encodings(encodings):
        for key, value in encodings.items():
            np.save(os.path.join('encoded_images', key), value)

    df = dedupe(df)
    batches = batch_df(df)

    for batch in tqdm(batches, total=len(df)//100):
        encodings = pipeline_batch(pipeline, batch)
        save_encodings(encodings)


def load_encodings(url):
    image_id = url.split('/')[-1]
    return np.load(os.path.join('encoded_images', image_id + '.npy'))

def preprocessed_generator(dataset, split, infinite_loop=False, batch_size=32, shuffle=True, **kwargs):
    X, y = dataset.get('X', split), dataset.get('y', split)

    dataset_size = X.shape[0]
    indices = X.index.tolist()

    if dataset_size == 0:  # Return None
        return

    first_run = True
    current_index = 0
    while True:
        if current_index == 0 and shuffle and not first_run:
            np.random.shuffle(indices)

        batch = indices[current_index:min(current_index + batch_size, dataset_size)]
        x_batch = X.loc[batch]
        y_batch = np.stack(y.loc[batch].squeeze().values)
        x_encodings = np.stack(x_batch.image.apply(load_encodings).values)
        x_captions = np.stack(x_batch.caption.values)
        yield [x_encodings, x_captions], y_batch

        current_index += batch_size

        # Loop so that infinite batches can be generated
        if current_index >= dataset_size:
            if infinite_loop:
                current_index = 0
                first_run = False
            else:
                break
