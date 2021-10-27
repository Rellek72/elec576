from imageio import imread
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np
# --------------------------------------------------
# Setup

learning_rate = 0.001
epochs = 10

ntrain = 1000   # per class
ntest = 100     # per class
nclass = 10     # number of classes
imsize = 28     # one dimension, 28x28
nchannels = 1
batchsize = 20

Train = np.zeros((ntrain*nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest*nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain*nclass, nclass))
LTest = np.zeros((ntest*nclass, nclass))

itrain = -1
itest = -1

for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot label
    for isample in range(0, ntest):
        path = 'CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1 # 1-hot label

# --------------------------------------------------
# model -- LeNet5
model = Sequential()

# Convolutional layer with kernel 5 x 5 and 32 filter maps followed by ReLU
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(imsize, imsize, nchannels)))
# Max Pooling layer subsampling by 2
model.add(MaxPool2D())
# Convolutional layer with kernel 5 x 5 and 64 filter maps followed by ReLU
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
# Max Pooling layer subsampling by 2
model.add(MaxPool2D())
# Fully Connected layer that has input 7*7*64 (3136) and output 1024
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
# Fully Connected layer that has input 1024 and output 10 (for the classes)
model.add(Dense(units=10, activation='softmax'))

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
checkpointer = ModelCheckpoint(filepath="weights.hdf5",
                               monitor = 'val_accuracy',
                               verbose=1,
                               save_best_only=True)
'''
# --------------------------------------------------
# optimization
training = model.fit(Train, LTrain, epochs=epochs, batch_size=batchsize, validation_data=(Test, LTest))

print(training.history.keys())
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('CNN Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('CNN Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# --------------------------------------------------
# test

score = model.evaluate(Test, LTest, batch_size=batchsize, verbose=2)

print("Test Loss: %f" % score[0])
print("Test Accuracy: %f" % score[1])

# --------------------------------------------------
# Visualization & Activation Metrics

model_activations = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
activations = model_activations.predict(Test)
type(activations)
type(activations)
type(activations)
type(activations)
type(activations)
type(activations)
type(activations)
type(activations)
type(activations)
type(activations)
for i in range(len(activations)):
    print("Activation", i, "mean:", np.mean(np.array(activations[i])), ", var:", np.var(np.array((activations[i]))))

for layer in model.layers:
    if 'conv' not in layer.name:
        print(layer.name)
        continue
    f, b = layer.get_weights()
    print(layer.name, f.shape)

filters, biases = model.layers[0].get_weights()
# Normalize
f_min = filters.min()
f_max = filters.max()
filters = (filters - f_min) / (f_max - f_min)

print(filters.shape)

n, m, k = 6, 3, 1
for i in range(n):
    f = filters[:, :, :, i]
    for j in range(m):
        print(f.shape)
        x = plt.subplot(n, m, k)
        x.set_xticks([])
        x.set_yticks([])
        plt.imshow(f, cmap='gray')
        k += 1
plt.show()
