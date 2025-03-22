# %%
!pip3 install tensorflow opencv-python matplotlib

# %%
import tensorflow as tf
import os

# %%
# Avoiding "out of memory" errors by setting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %%
!pip3 install cv2, imghdr

# %%
import cv2
import imghdr
from matplotlib import pyplot as plt

# %%
data_dir = 'data'
list = os.listdir(data_dir)
for item in list:
    if (item.startswith('.DS')):
        list.remove(item)

print(list)


# %%
img_exts = ['jpeg', 'jpg', 'bmp', 'png']

# %%
for img_class in list:
    img_class_path = os.path.join(data_dir, img_class)
    if os.path.isdir(img_class_path):
        for img in os.listdir(img_class_path):
            img_path = os.path.join(img_class_path, img)
            try:
                img = cv2.imread(img_path)
                tip = imghdr.what(img_path)
                if tip not in img_exts:
                    print('Image not in ext list {}'.format(img_path))
                    os.remove(img_path)
            except Exception as e:
                print('Issue with image {}'.format(img_path))


# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
data = tf.keras.utils.image_dataset_from_directory('data')

# %%
data_iterator = data.as_numpy_iterator()
data_iterator

# %%
# Images represented as numpy arrays
batch = data_iterator.next()
batch[0].shape

# %%
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

# %%
data = data.map(lambda x, y: (x/255.0, y))

# %%
scaled_iterator = data.as_numpy_iterator()

# %%
batch[0]

# %%
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# %%
len(data)

# %%

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data)* 0.1)+1

# %%
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# %%
model = Sequential()

# %%
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256,256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))



# %%
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# %%
model.summary()

# %%
logdir='logs'

# %%
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# %%
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# %%
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper right")
plt.show()

# %%
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal')
plt.plot(hist.history['val_accuracy'], color='orange')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# %%
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# %%
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# %%
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# %%
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

# %%
import cv2

# %%
img = cv2.imread('happytest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# %%
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

# %%
np.expand_dims(resize, 0).shape

# %%
yhat = model.predict(np.expand_dims(resize/255, 0))
yhat

# %%
if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')

# %%
from tensorflow.keras.models import load_model

# %%
model.save(os.path.join('models', 'happysadmodel.h5'))

# %%
os.path.join('models', 'happysadmodel.h5')

# %%
new_model = load_model(os.path.join('models', 'happysadmodel.h5'))

# %%
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

# %%
if yhatnew > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')

# %%



