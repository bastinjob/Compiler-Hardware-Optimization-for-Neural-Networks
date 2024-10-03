from models.model import create_model
from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype('float32') / 255
model = create_model()
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
model.save('models/unoptimized_model.h5')
