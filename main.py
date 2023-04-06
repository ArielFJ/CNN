import tensorflow as tf
import numpy as np

from keras import datasets, layers, models
import matplotlib.pyplot as plt

"""
El conjunto de datos CIFAR10 contiene 60 000 imágenes en color en 10 clases, con 6000 imágenes en cada clase.
El conjunto de datos se divide en 50 000 imágenes de entrenamiento y 10 000 imágenes de prueba.
Las clases son mutuamente excluyentes y no hay superposición entre ellas.
"""
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
print(np.argmax(test_labels))

# Normaliza los valores de los pixeles entre 0 y 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Verifica que el conjunto de datos sea correcto
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     # Los labels de CIFAR son arreglos, por eso se necesita el índice extra ([0])
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# Crea la base convolucional, una pila de capas Conv2D y MaxPooling2D
# Esto transformará el input en un tensor de 3 dimensiones (altura, ancho, canales)
# que sea más fácil de procesar que el input original de 4 dimensiones (imagenes)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Agrega capas densas en la parte superior para clasificar las imágenes
# transformando el tensor 3D en un tensor 1D
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# Compilar y entrenar modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluar modelo
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# Imprimir la precisión del modelo
print('Precisión del modelo:', test_acc)

# Haciendo las predicciones
pred = model.predict(test_images)
print(pred)

# Convirtiendo las predicciones en clases
pred_classes = np.argmax(pred, axis=1)
print(pred_classes)

# Graficando las predicciones
fig, axes = plt.subplots(5, 5, figsize=(15,15))
axes = axes.ravel()

for i in np.arange(0, 25):
    expected_index = i + 0 # Añade de 25 en 25 aquí para ver más imágenes
    axes[i].imshow(test_images[expected_index])
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[test_labels[expected_index][0]], class_names[pred_classes[expected_index]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)

plt.show()