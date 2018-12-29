'''
Module to define the pipeline(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.pipelines.dataset_pipelines.base_dataset_pipeline import BaseExplicitSplitDatasetPipeline
from simpleml.pipelines.production_pipelines.base_production_pipeline import BaseExplicitSplitProductionPipeline
from simpleml.pipelines.validation_split_mixins import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
from simpleml.utils.errors import PipelineError

import numpy as np
import pandas as pd


''' Dataset Pipelines '''

class UnsupervisedExplicitSplitDatasetPipeline(BaseExplicitSplitDatasetPipeline):
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
                    yield X.iloc[batch], np.stack(y.iloc[batch].squeeze().values)
                else:
                    yield X[batch], y[batch]
            else:
                if isinstance(X, (pd.DataFrame, pd.Series)):
                    yield X.iloc[batch], None
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


class NoFitExplicitSplitProductionPipeline(GeneratorPipeline, BaseExplicitSplitProductionPipeline):
    def __init__(self, *args, **kwargs):
        super(NoFitExplicitSplitProductionPipeline, self).__init__(*args, **kwargs)
        self.state['fitted'] = True
