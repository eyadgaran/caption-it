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
