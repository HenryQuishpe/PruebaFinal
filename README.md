# PruebaFinal
 Proyecto final tratamiento datos

#  Pre-requisitos

      _Jupyter notebook
      _GitHub

# 1.....index.ipynb
      Documento de trabajo de jupyter notebook donde generamos todo el codigo
# En este proyecto realizamos un modelo de clasificación de imágenes con importar las principales librerias:
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# cv2 que es la libreria de opencv para el procesamiento de imagenes.
# sklearn.preprocessing nos permite trabajar con las clases.
# keras nos permite entrenar el modelo
# os leemos el contenido de las imagenes
# numpy para el manejo de arreglos

# Definir las rutas de las carpetas de entrenamiento y prueba
train_folder = 'CarneDataset/train'
test_folder = 'CarneDataset/test'

# Leer las imágenes de entrenamiento y sus etiquetas
train_images = []
train_labels = []
for label in os.listdir(train_folder):
    label_folder = os.path.join(train_folder, label)
    for filename in os.listdir(label_folder):
        img_path = os.path.join(label_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            train_images.append(img)
            train_labels.append(label)

# Convertir las listas de imágenes y etiquetas a matrices NumPy
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Convertir las etiquetas en codificacion numerica para que python lo pueda interpretar internamente.
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

# Dividir el conjunto de entrenamiento en entrenamiento y validación
train_images, val_images, train_labels_encoded, val_labels_encoded = train_test_split(train_images, train_labels_encoded, test_size=0.2, random_state=42)

# Normalizar los valores de píxeles entre 0 y 1
train_images = train_images.astype('float32') / 255.0
val_images = val_images.astype('float32') / 255.0

# Definir la arquitectura de la CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train_images.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels_encoded, epochs=10, batch_size=32, validation_data=(val_images, val_labels_encoded))

# Leer las imágenes de prueba y sus etiquetas
test_images = []
test_labels = []
for label in os.listdir(test_folder):
    label_folder = os.path.join(test_folder, label)
    for filename in os.listdir(label_folder):
        img_path = os.path.join(label_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            test_images.append(img)
            test_labels.append(label)

# Convertir las listas de imágenes y etiquetas de prueba a matrices NumPy
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Normalizar los valores de píxeles de las imágenes de prueba entre 0 y 1
test_images = test_images.astype('float32') / 255.0

# Codificar las etiquetas de prueba como números
test_labels_encoded = label_encoder.transform(test_labels)

# Realizar las predicciones en el conjunto de prueba
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Calcular la matriz de confusión del modelo
confusion = confusion_matrix(test_labels_encoded, predicted_labels)
print('Matriz de confusión del modelo:')
print(confusion)

# Calcular la matriz de confusión del error en entrenamiento
train_predictions = model.predict(train_images)
train_predicted_labels = np.argmax(train_predictions, axis=1)
train_confusion = confusion_matrix(train_labels_encoded, train_predicted_labels)
print('Matriz de confusión del error en entrenamiento:')
print(train_confusion)

# Calcular la matriz de confusión del error en prueba
test_predictions = model.predict(test_images)
test_predicted_labels = np.argmax(test_predictions, axis=1)
test_confusion = confusion_matrix(test_labels_encoded, test_predicted_labels)
print('Matriz de confusión del error en prueba:')
print(test_confusion)
