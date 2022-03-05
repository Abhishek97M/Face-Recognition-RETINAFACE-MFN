import numpy as np
import tensorflow as tf
import time
 
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tflite models/tf_arcface_mobilefacenet_v1/tf_arcface_mobilefacenet_v1.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
while(True):
    t=time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    #print(input_data.shape)
    interpreter.invoke()
    print(output_details[0]['index'])

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data.shape)
    s=time.time()
    #print(str((s-t)*1000))
   
    
