'''
Module to define the pipeline(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.pipelines.dataset_pipelines.base_dataset_pipeline import BaseExplicitSplitDatasetPipeline
from simpleml.pipelines.production_pipelines.base_production_pipeline import BaseExplicitSplitProductionPipeline
from simpleml.pipelines.validation_split_mixins import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT


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

