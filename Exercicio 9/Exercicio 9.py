import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np


img_path = 'd:/Usuários/Felipe/Documents/Projetos/Mestrado/FIA/Exercicio 9/images/dog_1.jpg'
img_path_2 = 'd:/Usuários/Felipe/Documents/Projetos/Mestrado/FIA/Exercicio 9/images/lion.jpg'
img_path_3 = 'd:/Usuários/Felipe/Documents/Projetos/Mestrado/FIA/Exercicio 9/images/bear.jpg'

model_50 = ResNet50(weights='imagenet')
model_101 = ResNet101(weights='imagenet')
model_152 = ResNet152(weights='imagenet')

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

img = image.load_img(img_path_2, target_size=(224, 224))
x_1 = image.img_to_array(img)
x_1 = np.expand_dims(x_1, axis=0)
x_1 = preprocess_input(x_1)

img = image.load_img(img_path_3, target_size=(224, 224))
x_2 = image.img_to_array(img)
x_2 = np.expand_dims(x_2, axis=0)
x_2 = preprocess_input(x_2)


preds = model_50.predict(x)
print('Predicted Dog - ResNet50: ', decode_predictions(preds, top=3)[0])
preds = model_101.predict(x)
print('Predicted Dog - ResNet101:', decode_predictions(preds, top=3)[0])
preds = model_152.predict(x)
print('Predicted Dog - ResNet152:', decode_predictions(preds, top=3)[0])
print("")
preds_2 = model_50.predict(x_1)
print('Predicted Lion - ResNet50: ', decode_predictions(preds_2, top=3)[0])
preds_2 = model_101.predict(x_1)
print('Predicted Lion - ResNet101:', decode_predictions(preds_2, top=3)[0])
preds_2 = model_152.predict(x_1)
print('Predicted Lion - ResNet152:', decode_predictions(preds_2, top=3)[0])
print("")
preds_3 = model_50.predict(x_2)
print('Predicted Bear - ResNet50: ', decode_predictions(preds_3, top=3)[0])
preds_3 = model_101.predict(x_2)
print('Predicted Bear - ResNet101:', decode_predictions(preds_3, top=3)[0])
preds_3 = model_152.predict(x_2)
print('Predicted Bear - ResNet152:', decode_predictions(preds_3, top=3)[0])