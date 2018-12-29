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


class GeneratorKerasModel(BaseKerasModel):
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
        # This throws graph errors for some reason
        # self.external_model.fit_generator(
        #     generator=train_generator, validation_data=validation_generator, **params)

        epochs = params.get('epochs', 1)
        steps_per_epoch = params.get('steps_per_epoch', 1)
        validation_steps = params.get('validation_steps', 1)

        step = 0
        for epoch in range(epochs):
            LOGGER.info('EPOCH: {}'.format(epoch))
            for batch_x, batch_y in train_generator:
                self.external_model.fit(batch_x, batch_y)
                step += 1
                if step >= steps_per_epoch:
                    step = 0
                    break
            if validation_generator is not None:
                for batch_x, batch_y in validation_generator:
                    if len(batch_x) == 0:  # empty dataframe or ndarray
                        break
                    self.external_model.evaluate(batch_x, batch_y)
                    step += 1
                    if step >= validation_steps:
                        step = 0
                        break

    def _predict(self, X):
        '''
        Keras returns class tuples (proba equivalent) so cast to single prediction
        '''
        if isinstance(X, types.GeneratorType):
            return [self.external_model.predict(x) for x in X]

        return self.external_model.predict(X)


class ImageDecoder(GeneratorKerasModel):
    '''
    Networks used for training and predicting a seq2seq caption using
    an input image
    Dynamically creates inference network with training weights before predicting
    (real-time recurrent behavior is slightly different)
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

        ####################
        # Combined Decoder #
        ####################
        # lstm cell
        lstm = LSTM(LSTM_UNITS, return_sequences=True, name='decoder_lstm')(caption_embeddings,
                                                       initial_state=(initial_image_state, initial_image_state))

        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        lstm_bottleneck = TimeDistributed(Dense(LOGIT_BOTTLENECK, activation="elu", name='decoder_bottleneck'))(lstm)

        # logits bottleneck -> logits for next token prediction
        # Generate it for each timestamp independently
        next_token_prediction = Dense(VOCABULARY_SIZE, activation='softmax', name='prediction')(lstm_bottleneck)

        model = model([image_input, caption_input], next_token_prediction)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(),
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
        caption_input = Input(shape=(None,), dtype='int32', name='caption_input')

        # word -> embedding
        caption_embeddings = Embedding(VOCABULARY_SIZE, WORD_EMBED_SIZE, name='caption_embeddings')(caption_input)

        ####################
        # Combined Decoder #
        ####################
        # lstm cell
        lstm = LSTM(LSTM_UNITS, return_sequences=False, name='decoder_lstm')(caption_embeddings,
                                                       initial_state=(initial_image_state, initial_image_state))

        # we use bottleneck here to reduce model complexity
        # lstm output -> logits bottleneck
        lstm_bottleneck = Dense(LOGIT_BOTTLENECK, activation="elu", name='decoder_bottleneck')(lstm)

        # logits bottleneck -> logits for next token prediction
        # Generate it for each timestamp independently
        next_token_prediction = Dense(VOCABULARY_SIZE, activation='softmax', name='prediction')(lstm_bottleneck)

        model = model([image_input, caption_input], next_token_prediction)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=[])

        return model

    def transfer_weights(self, new_model, old_model):
        new_layers = {i.name: i for i in new_model.layers}
        old_layers = {i.name: i for i in old_model.layers}

        for name, layer in new_layers.items():
            if name in old_layers:
                layer.set_weights(old_layers[name].get_weights())

    def _predict(self, X):
        '''
        Inference network differs from training one so gets established dynamically
        at inference time. Does NOT get persisted since the weights are duplicative
        to the training ones. And the training network can in theory be updated
        with new training data later
        '''
        if not hasattr(self, 'inference_model'):
            self.inference_model = self.build_inference_network(WrappedKerasModel)
            self.transfer_weights(new_model=self.inference_model, old_model=self.external_model)

        if isinstance(X, types.GeneratorType):
            return [self.inference_model.predict(x) for x in X]

        return self.inference_model.predict(X)
