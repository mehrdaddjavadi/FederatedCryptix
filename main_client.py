import asyncio
from models.tensorflow_model import TensorFlowModel
from clients.client_device import ClientDevice

model = TensorFlowModel(model_config={'input_shape': [28, 28], 'layers': [{'type': 'Flatten', 'params': {}}, {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}}, {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}}], 'compile': {'optimizer': 'adam', 'loss': 'sparse_categorical_crossentropy', 'metrics': ['accuracy']}})
client = ClientDevice(client_id=1, model=model)

asyncio.run(client.connect_to_server('ws://localhost:8089'))
asyncio.run(client.receive_weights())
asyncio.run(client.train_model(x_train, y_train, training_config={'fit': {'epochs': 5, 'batch_size': 32}}))
asyncio.run(client.send_weights())
asyncio.run(client.send_data_request())
