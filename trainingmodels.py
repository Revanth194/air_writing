import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Part 1: Number Recognition Model using MNIST

# Load MNIST dataset for number recognition
(x_train_num, y_train_num), (x_test_num, y_test_num) = mnist.load_data()

# Preprocess the data
x_train_num = x_train_num.reshape(x_train_num.shape[0], 28, 28, 1).astype('float32') / 255
x_test_num = x_test_num.reshape(x_test_num.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to categorical one-hot encoding
num_classes_num = 10  # Digits 0-9
y_train_num = to_categorical(y_train_num, num_classes_num)
y_test_num = to_categorical(y_test_num, num_classes_num)

# Build the model
num_model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes_num, activation='softmax')
])

# Compile the model
num_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_model.fit(x_train_num, y_train_num, validation_data=(x_test_num, y_test_num), epochs=10, batch_size=200, verbose=2)

# Evaluate the model
score_num = num_model.evaluate(x_test_num, y_test_num, verbose=0)
print(f'Test loss for number model: {score_num[0]} / Test accuracy: {score_num[1]}')

# Save the model
num_model.save('bModel.h5')


# Part 2: Character Recognition Model using EMNIST


from tensorflow.keras.datasets import fashion_mnist as emnist

(x_train_char, y_train_char), (x_test_char, y_test_char) = emnist.load_data()

# Preprocess the data
x_train_char = x_train_char.reshape(x_train_char.shape[0], 28, 28, 1).astype('float32') / 255
x_test_char = x_test_char.reshape(x_test_char.shape[0], 28, 28, 1).astype('float32') / 255

# Convert labels to categorical one-hot encoding
num_classes_char = len(np.unique(y_train_char))
y_train_char = to_categorical(y_train_char, num_classes_char)
y_test_char = to_categorical(y_test_char, num_classes_char)

# Build the model
char_model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes_char, activation='softmax')
])

# Compile the model
char_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
char_model.fit(x_train_char, y_train_char, validation_data=(x_test_char, y_test_char), epochs=10, batch_size=200, verbose=2)

# Evaluate the model
score_char = char_model.evaluate(x_test_char, y_test_char, verbose=0)
print(f'Test loss for character model: {score_char[0]} / Test accuracy: {score_char[1]}')

# Save the model
char_model.save('bestmodel.h5')
