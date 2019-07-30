'''
Main module for "modeling" endpoints
'''

__author__ = 'Elisha Yadgaran'


from quart import request, render_template, flash, redirect, url_for

from backend.database.models import ModelHistory
from simpleml.utils import PersistableLoader
import base64
import pandas as pd
import numpy as np
import tensorflow as tf
import requests


class ModelWrapper(object):
    '''
    Lot of hackery to get the model to load in parallel when the service
    starts up

    Had trouble getting asyncio to actually execute in parallel so hacked the following:
    1) Load in thread
    2) Create new event loop for thread
    3) Save graph from thread to use in main thread at predict time
    '''
    def __init__(self):
        self._image_model = None
        self._text_model = None
        self._graph = None
        # self.concurrent_load_model()

    @property
    def image_model(self):
        if self._image_model is None:
            self.load_image_model()
        return self._image_model

    @property
    def text_model(self):
        if self._text_model is None:
            self.load_text_model()
        return self._text_model

    @property
    def graph(self):
        if self._graph is None:
            self.load_image_model()
        return self._graph

    def predict(self, image_source):
        with self.graph.as_default():
            X = pd.DataFrame({'image': image_source, 'caption': [self.text_model.initial_response]})
            tokens = self.image_model.predict(X, end_index=self.text_model.external_model.end_index, max_length=15)
            return self.text_model.inverse_transform(tokens[0])

    def load_image_model(self):
        self._image_model = PersistableLoader.load_model('image_model')
        self._image_model.load(load_externals=True)
        self._graph = tf.get_default_graph()

    def load_text_model(self):
        self._text_model = PersistableLoader.load_model('text_model')
        self._text_model.load(load_externals=True)

MODEL = ModelWrapper()


async def upload():
    if request.method == 'POST':  # For inputs with a binary image file
        files = await request.files
        if not 'photo' in files:
            raise ValueError('Missing photo')
        filename = files['photo'].filename
        image_stream = files['photo'].stream.read()

    elif request.method == 'GET':  # For inputs with an image url
        filename = request.args.get('url')
        image_stream = requests.get(filename, stream=True).raw.read()

    prediction = await predict(filename, image_stream)
    # .decode is necessary on python 3 for bytes to str conversion
    return await render_template(
        'pages/prediction.html',
        prediction=prediction.caption,
        image=base64.b64encode(image_stream).decode(),
        prediction_id=prediction.id
    )


async def predict(filename, image_stream):
    caption = MODEL.predict(image_stream)

    # DB
    history = ModelHistory.create(
        filename=filename,
        caption=caption
    )

    return history


async def model_feedback():
    form = await request.form
    prediction_id = form['prediction_id']
    user_rank = form['user_rank']
    user_caption = form['user_caption']
    history = ModelHistory.find(prediction_id)
    history.update(user_rank=user_rank, user_caption=user_caption)
    await flash("Thank you for making caption-bot smarter!")
    return redirect(url_for('home'))
