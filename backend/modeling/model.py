'''
Module to define the model(s) used
'''

__author__ = 'Elisha Yadgaran'


from backend.modeling.constants import SPECIAL_TOKEN_MAP, UNK_TOKEN

from simpleml.models.base_model import BaseModel
from simpleml.models.external_models import ExternalModelMixin
from simpleml.models.base_keras_model import BaseKerasModel

from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Masking, Input
from keras.optimizers import Adam
import numpy as np


class WrappedSklearnCountVectorizer(CountVectorizer, ExternalModelMixin):
    def get_index(self, token):
        if token in SPECIAL_TOKEN_MAP:
            return SPECIAL_TOKEN_MAP.get(token)
        return self.vocabulary_.get(token, SPECIAL_TOKEN_MAP[UNK_TOKEN])

    def get_token(self, index):
        if index in SPECIAL_TOKEN_MAP.values():
            return {v: k for k, v in SPECIAL_TOKEN_MAP.items()}.get(index)

        if not hasattr(self, 'reverse_vocab'):
            self.reverse_vocab = {v: k for k, v in self.vocabulary_.items()}

        return self.reverse_vocab.get(index)

    def predict(self, X):
        '''
        Assume X is an ndarray of tokens
        '''
        return np.apply_along_axis(self.index_tokens, 0, X)

    def index_tokens(self, token_list):
        '''
        Index list of tokens
        '''
        return [self.get_index(token) for token in token_list]

    def humanize_token_indices(self, index_list):
        '''
        Tokenize list of indices
        '''
        return ' '.join(
            self.get_token(index) for index in index_list
            if index not in SPECIAL_TOKEN_MAP.values()
        ).strip()


class TextProcessor(BaseModel):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnCountVectorizer(**kwargs)

    def inverse_tansform(self, *args):
        return self.external_model.humanize_token_indices(*args)

