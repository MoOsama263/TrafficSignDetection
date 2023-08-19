# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from tensorflow.keras.applications import VGG16
vgg16_obj = VGG16(include_top = False, input_shape = (224,224,3))    #  include_top = False is used to skip the layer from flattern
for layer in vgg16_obj.layers:             # Off the training of the trainable parameters
    layer.trainable = False
vgg16_obj.summary()
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Flatten
f1 = Flatten()(vgg16_obj.output)
final_layer = Dense(58, activation='softmax')(f1)
final_layer
model = Model(inputs=vgg16_obj.input,outputs=final_layer)
model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
traffic_datagen = ImageDataGenerator(rescale=1/255,
                                  shear_range=0.7,
                                  zoom_range=0.5)
path=r'C:/Users/MohamedKandil/PycharmProjects/GradiationProject/traffic_Data/DATA'
traffic_data =traffic_datagen.flow_from_directory(
    directory=path,
    target_size=(30,30),
    batch_size=3,
    class_mode="categorical",
    )

traffic_data.class_indices

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(traffic_data, epochs=10)

model.save('Trafffic_sign_prediction-98%.h5')