"""
convert_onnx.py

⚠️  WARNING: This script appears incomplete and contains syntax errors.

Intended to convert trained Keras/TensorFlow camera trap classification models to ONNX format
for deployment and cross-platform inference. The conversion uses tf2onnx to transform 
.h5 model files to .onnx format.

ISSUES FOUND:
- Syntax error: "from*keras" should be "from_keras"
- Incomplete unpacking: "onnx_model, * =" missing proper variable assignment
- Hardcoded file paths need to be parameterized

Requires fixing before use. Should include error handling and command-line arguments.
"""

import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

model = load_model('C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v10_20250325_170446/models/best_model_fox_v10_20250325_170446.h5')
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v10_20250325_170446/models/best_model_fox_v10_20250325_170446.onnx')
print("success")