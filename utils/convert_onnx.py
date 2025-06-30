import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

model = load_model('C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v10_20250325_170446/models/best_model_fox_v10_20250325_170446.h5')
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'C:/Users/c0062193.CAMPUS/OneDrive - Newcastle University/General - Fox-AI/AI results/Model performance/fox_v10_20250325_170446/models/best_model_fox_v10_20250325_170446.onnx')
print("success")