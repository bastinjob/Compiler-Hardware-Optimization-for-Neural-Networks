import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

def benchmark_model(model, test_data, batch_size=32):
    start_time = time.time()
    predictions = model.predict(test_data, batch_size=batch_size)
    end_time = time.time()
    
    return end_time - start_time, predictions

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255
    return x_test, y_test

def main():
    x_test, y_test = load_data()
    # Load the models
    unoptimized_model = load_model('models/unoptimized_model.h5')
    optimized_model_trt = load_model('models/optimized_model_trt/model.h5')  # Assuming you save it as a model
    optimized_model_tvm = load_model('models/optimized_model_tvm/model.h5')  # Assuming you save it as a model
    
    # Benchmark models
    for model in [unoptimized_model, optimized_model_trt, optimized_model_tvm]:
        elapsed_time, _ = benchmark_model(model, x_test)
        print(f"Elapsed time for {model.name}: {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    main()
