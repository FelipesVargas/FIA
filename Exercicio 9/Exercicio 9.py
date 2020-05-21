#import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np


img_path = 'd:/Usuários/Felipe/Documents/Projetos/Mestrado/FIA/Exercicio 9/images/dog_1.jpg'
model_50 = ResNet50(weights='imagenet')
model_152 = ResNet152V2(weights='imagenet')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model_152.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
# print: [[('n02109961', 'Eskimo_dog', 0.5123181), 
#          ('n02110185', 'Siberian_husky', 0.3543887), 
#          ('n02110063', 'malamute', 0.1273706), 


