# 4 parts
# 1 - Data preprocessing
# 2 - Building the CNN
# 3 - Training the CNN
# 4 - Making a single prediction
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

print(tf.__version__)

cnn = None

if os.path.exists('model.h5'):  # if model already exists
   print("Model file exists.")
   cnn = load_model('model.h5')

if cnn is None:  # if model is not created
    # Load and preprocess the training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        'training_set',
        image_size=(64, 64),
        batch_size=32,
        label_mode='binary'
    )
    # Print classes to check if they are correct
    print("Classes in train dataset:", train_dataset.class_names)

    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2),
        layers.RandomShear(0.2)
    ])

    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Normalization
    normalization_layer = layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Load and preprocess the test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        'test_set',
        image_size=(64, 64),
        batch_size=32,
        label_mode='binary',
        shuffle=False
    )
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))


    # Part 2 - Building the CNN

    cnn = Sequential([
        # First Convolutional Layer
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Second Convolutional Layer
        layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Flattening Layer
        layers.Flatten(),

        # Fully Connected Layer
        layers.Dense(units=128, activation='relu'),

        # Output Layer
        layers.Dense(units=1, activation='sigmoid')
    ])

    # Compile the CNN
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    # Part 3 - Training the CNN

    cnn.fit(train_dataset, validation_data=test_dataset, epochs=25)

    cnn.save('model.h5')
    print("Model saved")
    # Part 4 - Making a Single Prediction
else:
    print("Reusing the model")

test_image = image.load_img('prediction/k.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Ensure image is normalized

result = cnn.predict(test_image)

prediction = 'dog' if result[0][0] >= 0.5 else 'cat'
print(prediction)

