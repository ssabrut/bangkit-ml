import pandas as pd
import numpy as np
import tensorflow as tf

class PricePredictionModel:
    def __init__(self, data):
        self.data = data
        self.model = tf.keras.models.load_model('common/models/price_model.h5')

