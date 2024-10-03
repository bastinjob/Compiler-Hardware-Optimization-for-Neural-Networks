import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
import tensorflow as tf

def optimize_model_tvm(model):
    input_shape = (1, 28, 28, 1)  # Adjust based on your input shape
    model_input = np.random.rand(*input_shape).astype('float32')

    # Convert the Keras model to Relay IR
    mod, params = relay.frontend.from_keras(model)

    # Compile the model with TVM
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target="llvm", params=params)

    # Create a runtime executor
    module = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))

    # Set the input data
    module.set_input("input_1", model_input)
    
    # Run the model
    module.run()
    
    print("Model optimized with TVM.")

if __name__ == '__main__':
    model = tf.keras.models.load_model('models/unoptimized_model.h5')
    optimize_model_tvm(model)