import torch
import torch.nn as nn
import torch.optim as optim
from .abstract_model import AbstractModel

class PyTorchModel(AbstractModel):
    def __init__(self, model_config=None):
        self.model = self.create_model(model_config)

    def create_model(self, model_config):
        layers = []
        input_shape = model_config['input_shape'][0]
        for layer in model_config['layers']:
            layer_type = getattr(nn, layer['type'])
            if 'input_shape' in layer['params']:
                layers.append(layer_type(input_shape, **layer['params']))
                input_shape = layer['params']['out_features']
            else:
                layers.append(layer_type(**layer['params']))
        return nn.Sequential(*layers)

    def compile(self, compile_config):
        self.loss_fn = getattr(nn, compile_config['loss'])()
        self.optimizer = getattr(optim, compile_config['optimizer'])(self.model.parameters(), lr=compile_config['lr'])

    def train(self, x_train, y_train, training_config):
        self.model.train()
        for epoch in range(training_config['epochs']):
            self.optimizer.zero_grad()
            outputs = self.model(x_train)
            loss = self.loss_fn(outputs, y_train)
            loss.backward()
            self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def get_weights(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_weights(self, weights):
        for param, weight in zip(self.model.parameters(), weights):
            param.data = torch.tensor(weight, dtype=param.data.dtype)

    def predict(self, data):
        with torch.no_grad():
            return self.model(data).numpy()
