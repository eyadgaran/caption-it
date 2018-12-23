'''
Module to define the dataset(s) used for training and validation
'''

__author__ = 'Elisha Yadgaran'


from simpleml.datasets.processed_datasets.base_processed_dataset import BaseProcessedDataset, BasePandasProcessedDataset
from simpleml.datasets.raw_datasets.base_raw_dataset import BasePandasRawDataset
from simpleml.datasets.abstract_mixin import AbstractDatasetMixin
from simpleml.pipelines.validation_split_mixins import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

import pandas as pd
import requests
import zipfile
import StringIO
import json


COCO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
TRAIN_FILE = 'annotations/captions_train2017.json'
TEST_FILE = 'annotations/captions_val2017.json'


class MSCocoStreamingCaptionsRawDataset(BasePandasRawDataset):
    '''
    Use the Coco Dataset from Microsoft as a supervised caption source.
    Can download all the data locally and stream/load into memory, but given
    the size, this dataset will download images on the fly and store nothing
    locally.
    '''
    @staticmethod
    def download_metadata():
        # Download metadata zip
        zip_url = COCO_URL
        raw_response = requests.get(zip_url, stream=True)
        zip_stream = zipfile.ZipFile(StringIO.StringIO(raw_response.content))

        return zip_stream

    @staticmethod
    def load_metadata(zip_stream, filename):
        zip_info = [i for i in zip_stream.filelist if i.filename == filename][0]
        with zip_stream.open(zip_info) as f:
            stream = f.read()

        json_dict = json.loads(stream)
        images = pd.DataFrame(json_dict['images']).set_index('id', drop=False)

        def caption_agg(df):
            expanded = {'y_{}'.format(i): j for i, j in zip(range(len(df)), df.caption)}
            expanded.update({'caption_count': len(df)})
            return pd.Series(expanded)

        # Need to unstack because different size series are passed resulting in a pd.Series return
        captions = pd.DataFrame(json_dict['annotations']).groupby('image_id').apply(caption_agg).unstack()

        return images.join(captions, how='inner')

    def build_dataframe(self):
        '''
        Overwrite base method to not require raw datasets/dataset pipelines
        '''
        zip = self.download_metadata()

        self._external_file = {
            TRAIN_SPLIT: self.load_metadata(zip, TRAIN_FILE),
            VALIDATION_SPLIT: None,
            TEST_SPLIT: self.load_metadata(zip, TEST_FILE)
        }

    def get_feature_names(self):
        return self.get('X', TRAIN_SPLIT).columns.tolist()

