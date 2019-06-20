import os
from tensorflow.python import pywrap_tensorflow

model_dir = './checkpoints/accheckpoints/'
checkpoint_path = os.path.join(model_dir, "model.ckpt-46800.meta")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))