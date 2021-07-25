from data import load_img,data_augment
from augment import data_augment
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# Import Data

data_dir = pathlib.Path("/dataset/")
paths = list(data_dir.glob('*/*.jpg'))
def load_paths(path):
    path = path.absolute().as_posix()
    if "Fresh" in path:return str(path),1
    else:return path,0
ds = pd.DataFrame(map(load_paths,paths),columns =['Path','Fresh']).sample(frac=1).reset_index(drop=True)

# Keeping Out 16 images to test Later

ds = ds[:-16]
test = ds[-16:]

# Load images to tf.dataset

dataset = tf.data.Dataset.from_tensor_slices((ds.Path.values,ds.Fresh.values))

train_ds = dataset.take(int(0.8*len(ds)))
val_ds = dataset.skip(int(0.2*len(ds)))
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(load_img,num_parallel_calls=AUTOTUNE).repeat(4).map(data_augment,num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.map(load_img,num_parallel_calls=AUTOTUNE).batch(32).prefetch(buffer_size=AUTOTUNE)

# Define Model

base_model = keras.applications.VGG19(weights="imagenet",include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(2, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

# Train Model

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history = model.fit(train_ds, epochs=5, validation_data=val_ds,verbose=1)

# Plotting Metrics

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(history.history['val_loss']))
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Testing

test_ds = tf.data.Dataset.from_tensor_slices((test.Path.values))
test_ds = test_ds.map(load_test_img).batch(3)
prediction = model.predict(test_ds)

plt.figure(figsize=(15,15))
i=0
for index,row in test.iterrows():
    img = tf.io.decode_jpeg(tf.io.read_file(row['Path']),channels=3)
    plt.subplot(4,4,i+1)
    plt.imshow(img.numpy().astype("uint8"),aspect='auto')
    if prediction[i][0]>prediction[i][1]:
        if row['Fresh']==0:plt.title("Not Fresh", color="green")
        else:plt.title("Not Fresh", color="red")
    else:
        if row['Fresh']==0:plt.title("Fresh", color="red")
        else:plt.title("Fresh", color="green")
    plt.xticks([]), plt.yticks([])
    i+=1    

plt.show()    

# Save model for later use

model.save("model.h5")