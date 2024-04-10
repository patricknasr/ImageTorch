import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# Load labels
labels_df = pd.read_csv('training_data/Labels-IsEpicIntro-2024-03-25.csv')
labels_df['Is Epic'] = labels_df['Is Epic'].astype(str) # Convert labels to binary

# Split data into train and test sets
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Image paths and labels
train_filenames = train_df['Filename'].values
train_labels = train_df['Is Epic'].values
test_filenames = test_df['Filename'].values
test_labels = test_df['Is Epic'].values

# ImageDataGenerator for data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Because we're using ImageDataGenerator, we need a directory. Assuming spectrogram images are stored in 'training_data' directory.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='./training_data',
    x_col="Filename",
    y_col="Is Epic",
    target_size=(900, 1200),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='./training_data',
    x_col="Filename",
    y_col="Is Epic",
    target_size=(900, 1200),
    batch_size=32,
    class_mode='binary')

# Model architecture
model = tf.keras.Sequential([
    tf.keras.Input(shape=(900, 1200, 3)),  # Use tf.keras.Input to specify the input shape
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=test_generator, epochs=10, verbose=1)

# Model summary
model.summary()