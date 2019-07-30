'''
Module to define the pipeline(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml.pipelines import ExplicitSplitPipeline, Split, TransformedSequence
from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

import numpy as np
import pandas as pd
import os
from tqdm import tqdm


''' Dataset Pipelines '''

class UnsupervisedExplicitSplitPipeline(ExplicitSplitPipeline):
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
        self._dataset_splits = self.containerize_split({
            TRAIN_SPLIT: Split(X=self.dataset.get('X', TRAIN_SPLIT).join(self.dataset.get('y', TRAIN_SPLIT))),
            VALIDATION_SPLIT: Split(X=self.dataset.get('X', VALIDATION_SPLIT).join(self.dataset.get('y', VALIDATION_SPLIT))),
            TEST_SPLIT: Split(X=self.dataset.get('X', TEST_SPLIT).join(self.dataset.get('y', TEST_SPLIT)))
        })


class PreprocessedPipeline(ExplicitSplitPipeline):
    '''
    Custom pipeline class that overwrites default _iterate_dataset method
    to load pretransformed encodings from disk instead of transforming in real
    time. This is "safe" because generator use is only allowed during training;
    production runtime will be unaffected by this optimization
    '''
    def __init__(self, encoded_image_path='encoded_images', *args, **kwargs):
        super(PreprocessedPipeline, self).__init__(*args, **kwargs)
        self.state['encoded_image_path'] = encoded_image_path

    def fit(self):
        '''
        Extend parent routine with a step to pre-encode all the images
        '''
        # Fit the pipeline, if need be
        super(PreprocessedPipeline, self).fit()

        # Then transform all images and save to disk
        if not self.state.get('images_encoded', None):
            self.encode_all_images(self.X(TRAIN_SPLIT))
            self.encode_all_images(self.X(VALIDATION_SPLIT))
            self.state['images_encoded'] = True

        return self

    def encode_all_images(self, df):
        '''
        Preprocess for training speed
        Map image files to preencoded ndarrays (dont do direct because mutliple indices map to the same image)

        Outputs to disk a directory with every file named {image_name}.npy
        '''
        df = self.dedupe(df)
        batches = self.batch_df(df)

        for batch in tqdm(batches, total=len(df)//100):
            encodings = self.pipeline_batch(batch)
            self.save_encodings(encodings)

    def dedupe(self, df):
        # 1) Dedupe repeated images
        df.drop_duplicates(subset='image', keep='first', inplace=True)
        df['image_id'] = df.image.apply(lambda row: row.split('/')[-1])
        df.set_index('image_id', inplace=True)

        # Support resuming if operation crashes
        existing = [i[:-4] for i in os.listdir(self.state['encoded_image_path']) if i[-3:] == 'npy']
        overlapping_existing = [i for i in existing if i in df.index]
        print("Images to encode: {}, new images: {}".format(len(df), len(existing) - len(overlapping_existing)))
        df.drop(overlapping_existing, axis=0, inplace=True)

        return df

    @staticmethod
    def batch_df(df, batch_size=100):
        # 2) Batch process
        df_size = len(df)
        indices = df.index.tolist()
        current_index = 0

        while current_index < df_size:
            next_batch_size = min(batch_size, df_size - current_index)
            next_indices = indices[current_index: current_index + next_batch_size]
            yield df.loc[next_indices]
            current_index += next_batch_size

    def pipeline_batch(self, batch):
        # 3) Run through pipeline
        batch_indices = batch.index.tolist()
        transformed_batch = next(self.transform(batch))[0]
        return dict(zip(batch_indices, transformed_batch))

    def save_encodings(self, encodings):
        # 4) Save encoded images
        for key, value in encodings.items():
            np.save(os.path.join(self.state['encoded_image_path'], key), value)

    def load_encodings(self, url):
        image_id = url.split('/')[-1]
        return np.load(os.path.join(self.state['encoded_image_path'], image_id + '.npy'))

    def _generator_transform(self, X, dataset_split=None, **kwargs):
        '''
        Overwrite normal transform method to load from disk instead

        Pass through method to external pipeline

        :param X: dataframe/matrix to transform, if None, use internal dataset

        NOTE: Downstream objects expect to consume a generator with a tuple of
        X, y, other... not a Split object, so an ordered tuple will be returned
        '''
        if X is None:
            generator_split = self.get_dataset_split(dataset_split, return_generator=True, **kwargs)
            for batch in generator_split:  # Return is a generator of Split objects
                x_encodings = np.stack(batch.X.image.apply(self.load_encodings).values)
                x_captions = np.stack(batch.X.caption.values)
                output = [x_encodings, x_captions]

                # Return input with X replaced by output (transformed X)
                # Contains y or other named inputs to propagate downstream
                # Explicitly order for *args input -- X, y, other...
                return_objects = [output]
                if batch.y is not None:
                    return_objects.append(batch.y)
                for k, v in batch.items():
                    if k not in ('X', 'y'):
                        return_objects.append(v)
                yield tuple(return_objects)

        else:
            yield self.external_pipeline.transform(X, **kwargs)

    def _sequence_transform(self, X, dataset_split=None, **kwargs):
        '''
        Overwrite normal transform method to load from disk instead of transforming on the fly

        :param X: dataframe/matrix to transform, if None, use internal dataset

        NOTE: Downstream objects expect to consume a sequence with a tuple of
        X, y, other... not a Split object, so an ordered tuple will be returned
        '''
        if X is None:
            dataset_sequence = self.get_dataset_split(dataset_split, return_sequence=True, **kwargs)
            return PreloadedSequence(self, dataset_sequence)

        else:
            return self.external_pipeline.transform(X, **kwargs)


class PreloadedSequence(TransformedSequence):
    def __getitem__(self, *args, **kwargs):
        '''
        Pass-through to dataset sequence - applies transform on raw data and returns batch
        '''
        raw_batch = self.dataset_sequence.__getitem__(*args, **kwargs)  # Split object

        x_encodings = np.stack(raw_batch.X.image.apply(self.pipeline.load_encodings).values)
        x_captions = np.stack(raw_batch.X.caption.values)
        transformed_batch = [x_encodings, x_captions]

        # Return input with X replaced by output (transformed X)
        # Contains y or other named inputs to propagate downstream
        # Explicitly order for *args input -- X, y, other...
        return_objects = [transformed_batch]
        if raw_batch.y is not None:
            return_objects.append(raw_batch.y)
        for k, v in raw_batch.items():
            if k not in ('X', 'y'):
                return_objects.append(v)
        return tuple(return_objects)
