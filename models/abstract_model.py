import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from .abstract_model import AbstractModel

class TensorFlowModel(AbstractModel):
    def __init__(self, model_config=None):
        self.model = self.create_model(model_config)

    def create_model(self, model_config):
        model = Sequential()
        for layer in model_config['layers']:
            layer_type = getattr(tf.keras.layers, layer['type'])
            model.add(layer_type(**layer['params']))
        model.compile(**model_config['compile'])
        return model

    def train(self, x_train, y_train, training_config):
        self.model.fit(x_train, y_train, **training_config['fit'])

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
    
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def predict(self, data):
        return self.model.predict(data)
