'''
Module to define the model(s) used
'''

__author__ = 'Elisha Yadgaran'


from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT
from simpleml.models import BaseModel, KerasEncoderDecoderClassifier
from simpleml.models.external_models import ExternalModelMixin
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


class GeneratorKerasEncoderDecoder(KerasEncoderDecoderClassifier):
    '''
    Extends Base Keras Model to support data generators for fitting

    ONLY use with supporting pipeline!
    '''
    def fit(self, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        '''
        if self.pipeline is None:
            raise ModelError('Must set pipeline before fitting')

        if self.state['fitted']:
            LOGGER.warning('Cannot refit model, skipping operation')
            return self

        # Explicitly fit only on train split
        train_generator = self.pipeline.transform(X=None, dataset_split=TRAIN_SPLIT, return_y=True, infinite_loop=True, **self.get_params())
        validation_generator = self.pipeline.transform(X=None, dataset_split=VALIDATION_SPLIT, return_y=True, infinite_loop=True, **self.get_params())

        self._fit(train_generator, validation_generator)

        # Mark the state so it doesnt get refit and can now be saved
        self.state['fitted'] = True

        return self

    def _fit(self, train_generator, validation_generator=None):
        '''
        Keras fit parameters (epochs, callbacks...) are stored as self.params so
        retrieve them automatically
        '''
        # Generator doesnt take arbitrary params so pop the extra ones
        extra_params = ['batch_size']
        params = {k:v for k, v in self.get_params().items() if k not in extra_params}
        self.external_model.fit_generator(
            generator=train_generator, validation_data=validation_generator, **params)

    def _predict(self, X, end_index, max_length=None, **kwargs):
        '''
        Inference network differs from training one so gets established dynamically
        at inference time. Does NOT get persisted since the weights are duplicative
        to the training ones. And the training network can in theory be updated
        with new training data later

        Runs full encoder/decoder loop
        1) Encodes input into initial decoder state
        2) Loops through decoder state until:
            - End token is predicted
            - Max length is reached

        Generator Pipeline yields batches so yield batches as well
        '''
        self.check_for_models()

        # X is a generator tuple of encoder_input, decoder_input
        for batch in X:
            encoder_input, predicted_sequence = batch

        # Encode the input as state vectors for the decoder.
        encoder_states = self.encode(encoder_input)
        decoder_states = [encoder_states, encoder_states]  # GRU uses single state vs lstm's h, c

        while True:
            decoder_output_and_states = self.decode([predicted_sequence] + decoder_states)
            decoder_prediction = decoder_output_and_states[0]
            decoder_states = decoder_output_and_states[1:]  # Multiple for lstm, single for gru

            # Next tokens
            next_tokens = np.argmax(decoder_prediction, axis=1).reshape(-1, 1)
            if len(predicted_sequence.shape) == 1:  # 1d array
                axis = 0
            else:
                axis = 1
            predicted_sequence = np.concatenate([predicted_sequence, next_tokens], axis=axis)

            # Exit conditions
            # TODO: generic support for different length sequences
            if (next_tokens == end_index).any() or (max_length is not None and predicted_sequence.shape[axis] >= max_length):
                break

        return predicted_sequence


class ImageDecoder(GeneratorKerasEncoderDecoder):
    '''
    Networks used for training and predicting a seq2seq caption using
    an input image
    Dynamically creates inference network with training weights before predicting
    (real-time recurrent behavior is slightly different)
    '''
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
        CAPTION_LENGTH = kwargs.get('pad_length') - 1  # Substract one because we dont predict the start token
        PAD_INDEX = kwargs.get('pad_index')

        # Save config values for later
        new_configs =  {
                'IMG_EMBED_SIZE': IMG_EMBED_SIZE,
                'IMG_EMBED_BOTTLENECK': IMG_EMBED_BOTTLENECK,
                'WORD_EMBED_SIZE': WORD_EMBED_SIZE,
                'LSTM_UNITS': LSTM_UNITS,
                'LOGIT_BOTTLENECK': LOGIT_BOTTLENECK,
                'VOCABULARY_SIZE': VOCABULARY_SIZE,
                'CAPTION_LENGTH': CAPTION_LENGTH,
                'PAD_INDEX': PAD_INDEX,
            }
        self.config.update(new_configs)

        ###############
        # Image Input #
        ###############
        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        image_input = Input(shape=(IMG_EMBED_SIZE,), dtype='float32', name='image_input')

        # we use bottleneck here to reduce the number of parameters
        # image embedding -> bottleneck
        img_bottleneck = Dense(IMG_EMBED_BOTTLENECK, activation='elu', name='image_bottleneck')(image_input)

        # image embedding bottleneck -> lstm initial state
        initial_image_state = Dense(LSTM_UNITS, activation='elu', name='image_context')(img_bottleneck)

        #################
        # Caption Input #
        #################
        # [batch_size, time steps] of word ids
        caption_input = Input(shape=(CAPTION_LENGTH,), dtype='int32', name='caption_input')

        # Mask padding
        padding_mask = Masking(mask_value=PAD_INDEX)(caption_input)

        # word -> embedding
        caption_embeddings = Embedding(VOCABULARY_SIZE, WORD_EMBED_SIZE, name='caption_embedding')(padding_mask)

        ###########
        # Decoder #
        ###########
        # lstm cell
        decoder_output = LSTM(LSTM_UNITS, return_sequences=True, name='decoder_lstm')(caption_embeddings,
                              initial_state=(initial_image_state, initial_image_state))

        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        decoder_bottleneck = Dense(LOGIT_BOTTLENECK, activation="elu", name='decoder_bottleneck')(decoder_output)

        # logits bottleneck -> logits for next token prediction
        # Generate it for each timestamp independently
        next_token_prediction = Dense(VOCABULARY_SIZE, activation='softmax', name='prediction')(decoder_bottleneck)

        model = model([image_input, caption_input], next_token_prediction)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=[])

        # print(model.summary())
        # from keras.utils.vis_utils import plot_model
        # plot_model(model, to_file='model.png', show_shapes=True)

        return model

    def build_inference_network(self, model):
        '''
        Inference network - Differs from training one so gets established dynamically
        at inference time

        Input:
            X = [image embedding, tokenized_caption]
            y = [shifted_tokenized_caption]

        Output:
            y = [predicted_tokenized_captions]
        '''
        IMG_EMBED_SIZE = self.config.get('IMG_EMBED_SIZE')
        IMG_EMBED_BOTTLENECK = self.config.get('IMG_EMBED_BOTTLENECK')
        WORD_EMBED_SIZE = self.config.get('WORD_EMBED_SIZE')
        LSTM_UNITS = self.config.get('LSTM_UNITS')
        LOGIT_BOTTLENECK = self.config.get('LOGIT_BOTTLENECK')
        VOCABULARY_SIZE = self.config.get('VOCABULARY_SIZE')

        #################
        # Encoder Input #
        #################
        # [batch_size, IMG_EMBED_SIZE] of CNN image features
        image_input = Input(shape=(IMG_EMBED_SIZE,), dtype='float32', name='image_input')

        # we use bottleneck here to reduce the number of parameters
        # image embedding -> bottleneck
        img_bottleneck = Dense(IMG_EMBED_BOTTLENECK, activation='elu', name='image_bottleneck')(image_input)

        # image embedding bottleneck -> lstm initial state
        initial_image_state = Dense(LSTM_UNITS, activation='elu', name='image_context')(img_bottleneck)

        encoder_model = model(image_input, initial_image_state)
        encoder_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam')

        #################
        # Decoder Input #
        #################
        # [batch_size, time steps] of word ids
        caption_input = Input(shape=(None,), dtype='int32', name='caption_input')
        decoder_state_h_input = Input(shape=(LSTM_UNITS,), dtype='float32', name='decoder_h_input')
        decoder_state_c_input = Input(shape=(LSTM_UNITS,), dtype='float32', name='decoder_c_input')

        # word -> embedding
        caption_embeddings = Embedding(VOCABULARY_SIZE, WORD_EMBED_SIZE, name='caption_embeddings')(caption_input)

        # lstm cell
        decoder_output, state_h, state_c = LSTM(LSTM_UNITS, return_state=True, name='decoder_lstm')(caption_embeddings,
                                                initial_state=[decoder_state_h_input, decoder_state_c_input])

        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        decoder_bottleneck = Dense(LOGIT_BOTTLENECK, activation="elu", name='decoder_bottleneck')(decoder_output)

        # logits bottleneck -> logits for next token prediction
        next_token_prediction = Dense(VOCABULARY_SIZE, activation='softmax', name='prediction')(decoder_bottleneck)

        decoder_model = model([caption_input, decoder_state_h_input, decoder_state_c_input], [next_token_prediction, state_h, state_c])
        decoder_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam')

        return encoder_model, decoder_model
