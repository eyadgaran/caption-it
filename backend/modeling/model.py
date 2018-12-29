'''
Module to define the model(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT
from simpleml.models.base_model import BaseModel
from simpleml.models.external_models import ExternalModelMixin
from simpleml.models.base_keras_model import BaseKerasModel
from simpleml.utils.errors import ModelError

from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Masking, Input
from keras.optimizers import Adam
import numpy as np
import logging
import types


LOGGER = logging.getLogger(__name__)


class WrappedSklearnCountVectorizer(CountVectorizer, ExternalModelMixin):
    def __init__(self, pad_token="#PAD#", unknown_token="um",
                 start_token="#START#", end_token="#END#", **kwargs):
        super(WrappedSklearnCountVectorizer, self).__init__(**kwargs)
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token
        self.pad_index = 0
        self.start_index = 1
        self.unknown_index = 2
        self.end_index = 3

    def get_index(self, token):
        return self.vocabulary_.get(token, self.vocabulary_.get(self.unknown_token))

    def get_token(self, index):
        if not hasattr(self, 'reverse_vocab'):
            self.reverse_vocab = {v: k for k, v in self.vocabulary_.items()}

        return self.reverse_vocab.get(index)

    def fit(self, *args, **kwargs):
        super(WrappedSklearnCountVectorizer, self).fit(*args, **kwargs)
        vocab = self.vocabulary_
        special_tokens = [self.pad_token, self.start_token, self.unknown_token, self.end_token]
        for token in special_tokens:
            vocab.pop(token, None)
        self.vocabulary_ = {token: index for index, token in enumerate(special_tokens + vocab.keys())}

        # sanity check
        for token, index in zip(special_tokens, [self.pad_index, self.start_index, self.unknown_index, self.end_index]):
            assert(index == self.vocabulary_.get(token))

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
            if index not in [self.pad_index, self.start_index, self.end_index]
        ).strip()


class TextProcessor(BaseModel):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnCountVectorizer(**kwargs)

    def inverse_transform(self, *args):
        return self.external_model.humanize_token_indices(*args)

    @property
    def initial_response(self):
        '''
        When a request first comes in, only the start token is available.
        Subsequent output tokens are recursively fed back into the neural net
        '''
        return np.array([self.external_model.start_index])


class WrappedKerasModel(Model, ExternalModelMixin):
    pass


class TrainingImageDecoder(BaseKerasModel):
    '''
    Network used for training only (real-time recurrent behavior is slightly different)
    '''
    def _create_external_model(self, **kwargs):
        external_model = WrappedKerasModel
        return self.build_network(external_model, **kwargs)

    def build_network(self, model, **kwargs):
        '''
        training network

        Input:
            X = [image embedding, tokenized_caption]
            y = [shifted_tokenized_caption]

        Output:
            y = [predicted_tokenized_captions]
        '''
        IMG_EMBED_SIZE = 2048  # InceptionV3 output
        IMG_EMBED_BOTTLENECK = 120
        WORD_EMBED_SIZE = 100
        LSTM_UNITS = 300
        LOGIT_BOTTLENECK = 120
        VOCABULARY_SIZE = kwargs.get('vocabulary_size')
        CAPTION_LENGTH = kwargs.get('pad_length')
        PAD_INDEX = kwargs.get('pad_index')

        ###############
        # Image Input #
        ###############
        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        image_input = Input(shape=(IMG_EMBED_SIZE,), dtype='float32')

        # we use bottleneck here to reduce the number of parameters
        # image embedding -> bottleneck
        img_bottleneck = Dense(IMG_EMBED_BOTTLENECK, activation='elu')(image_input)

        # image embedding bottleneck -> lstm initial state
        initial_image_state = Dense(LSTM_UNITS, activation='elu')(img_bottleneck)

        #################
        # Caption Input #
        #################
        # [batch_size, time steps] of word ids
        caption_input = Input(shape=(CAPTION_LENGTH,), dtype='int32')

        # Mask padding
        padding_mask = Masking(mask_value=PAD_INDEX)(caption_input)

        # word -> embedding
        caption_embeddings = Embedding(VOCABULARY_SIZE, WORD_EMBED_SIZE)(padding_mask)

        ####################
        # Combined Decoder #
        ####################
        # lstm cell
        lstm = LSTM(LSTM_UNITS, return_sequences=True)(caption_embeddings,
                                                       initial_state=(initial_image_state, initial_image_state))

        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        lstm_bottleneck = Dense(LOGIT_BOTTLENECK, activation="elu")(lstm)

        # logits bottleneck -> logits for next token prediction
        # Generate it for each timestamp independently
        next_token_prediction = TimeDistributed(Dense(VOCABULARY_SIZE))(lstm_bottleneck)

        model = model([image_input, caption_input], next_token_prediction)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        print(model.summary())
        # from keras.utils.vis_utils import plot_model
        # plot_model(model, to_file='model.png', show_shapes=True)

        return model
