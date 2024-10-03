import tensorflow as tf
import tensorflow_datasets as tfds
import  tensorflow_model_optimization as tmot

def optimize_model(model):

    model.save('models/optimized_model_trt/model')
    saved_model_loaded = tf.saved_model.load('models/optimized_model_trt/model')

    converter = tf.experimental.tensorrt.Converter(input_saved_model_dir='models/optimized_model_trt/model')
    converter.convert()

    converter.save('models/optimized_model_trt/model_trt')

    print('Model optimized with TensorRT')


    if __name__ == '__main__':
        model = tf.keras.models.load_model('models/unoptimized_model.h5')
        optimize_model(model)