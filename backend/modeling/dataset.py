'''
Module to define the dataset(s) used for training and validation
'''

__author__ = 'Elisha Yadgaran'


from simpleml.datasets import BaseDataset, BasePandasDataset
from simpleml.datasets.abstract_mixin import AbstractDatasetMixin
from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

import pandas as pd
import requests
import zipfile

# Python 2/3 compatibility
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import json
from sklearn.model_selection import train_test_split


COCO_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
TRAIN_FILE = 'annotations/captions_train2017.json'
TEST_FILE = 'annotations/captions_val2017.json'


class MSCocoStreamingCaptionsRawDataset(BasePandasDataset):
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
        zip_stream = zipfile.ZipFile(StringIO(raw_response.content))

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


class MSCocoCaptionsDataset(BaseDataset, AbstractDatasetMixin):
    '''
    Processed unsupervised dataset with only captions
    Can be used for tokenization and embeddings
    '''
    def build_dataframe(self):
        '''
        Overwrite base method to not require raw datasets/dataset pipelines
        '''
        self._external_file = {
            TRAIN_SPLIT: self.pipeline.transform(X=None, dataset_split=TRAIN_SPLIT, return_y=True),
            VALIDATION_SPLIT: self.pipeline.transform(X=None, dataset_split=VALIDATION_SPLIT, return_y=True),
            TEST_SPLIT: self.pipeline.transform(X=None, dataset_split=TEST_SPLIT, return_y=True)
        }

    def get(self, column, split):
        if column not in ('X', 'y'):
            raise ValueError('Only support columns: X & y')
        if split not in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT):
            raise ValueError('Only support splits: {}, {}, {}'.format(
                TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT))

        x, y = self.dataframe.get(split)
        if x is None:
            x = pd.DataFrame()
        if y is None:
            y = pd.DataFrame()

        if column == 'y':
            return y

        else:
            return x

    def get_feature_names(self):
        return ['X']


class MSCocoStreamingCaptionsEncodedDataset(BasePandasDataset, AbstractDatasetMixin):
    '''
    Use the Coco Dataset from Microsoft as a supervised caption source.
    Can download all the data locally and stream/load into memory, but given
    the size, this dataset will download images on the fly and store nothing
    locally.

    Format:
    X                                    |   y
    -------------------------------------------------------------------
    url     |   Tokenized caption        |   Offset tokenized caption
    -------------------------------------------------------------------
    http:...|   [START,FIRST...END..PAD] |  [FIRST...END...PAD]
    '''
    def build_dataframe(self):
        '''
        Overwrite base method to not require raw datasets/dataset pipelines
        Also manually create a validation set (default doesnt have one)
        '''
        X, y = self.pipeline.transform(X=None, dataset_split=TRAIN_SPLIT, return_y=True)
        df = pd.concat((X, y), axis=1)
        train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

        self._external_file = {
            TRAIN_SPLIT: train_df,
            VALIDATION_SPLIT: validation_df,
            TEST_SPLIT: pd.concat(self.pipeline.transform(X=None, dataset_split=TEST_SPLIT, return_y=True), axis=1)
        }

        if not self.label_columns:  # Skip if explicitly passed to constructor
            if y is None:
                y = pd.DataFrame()
            self.config['label_columns'] = y.columns.tolist()

    def get_feature_names(self):
        return self.get('X', TRAIN_SPLIT).columns.tolist()
