import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)



class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.90):
            print("\nReached 90% accuracy, ending training.")
            self.model.stop_training = True
            

callbacks= myCallback()
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()


#plt.imshow(training_images[0])
#print(training_labels[0])
#print(training_images[0])

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images/ 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

print("variables normalized")
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

print("model created")

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("model compiled")

model.summary()

model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

print("model fit")

#test_loss, test_acc = model.evaluate(test_images, test_labels)
model.evaluate(test_images,test_labels)
print("model evaled") 

predictions = model.predict(test_images)

print(predictions[0])

print(test_labels[0])
