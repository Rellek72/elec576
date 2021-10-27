import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as KL
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist
(Train, LTrain), (Test, LTest) = mnist.load_data()

learningRate = 0.01
epochs = 10  # equivilant to trainingIters/len(Train)
batchSize = 50

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 124 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

# RNN
model_rnn = Sequential()
model_rnn.add(KL.SimpleRNN(nHidden, input_shape=(nInput, nSteps)))
model_rnn.add(KL.Dense(nClasses, activation='softmax'))
model_rnn.compile(optimizer=optimizers.Adam(learning_rate=learningRate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# LSTM
model_lstm = Sequential()
model_lstm.add(KL.LSTM(nHidden, input_shape=(nInput, nSteps)))
model_lstm.add(KL.Dense(nClasses, activation='softmax'))
model_lstm.compile(optimizer=optimizers.Adam(learning_rate=learningRate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# GRU
model_gru = Sequential()
model_gru.add(KL.GRU(nHidden, input_shape=(nInput, nSteps)))
model_gru.add(KL.Dense(nClasses, activation='softmax'))
model_gru.compile(optimizer=optimizers.Adam(learning_rate=learningRate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TRAINING
training_rnn = model_rnn.fit(Train, LTrain, epochs=epochs, batch_size=batchSize, validation_data=(Test, LTest))
training_lstm = model_lstm.fit(Train, LTrain, epochs=epochs, batch_size=batchSize, validation_data=(Test, LTest))
training_gru = model_gru.fit(Train, LTrain, epochs=epochs, batch_size=batchSize, validation_data=(Test, LTest))

# PLOTTING ACCURACY
plt.plot(training_rnn.history['accuracy'])
plt.plot(training_lstm.history['accuracy'])
plt.plot(training_gru.history['accuracy'])
plt.plot(training_rnn.history['val_accuracy'])
plt.plot(training_lstm.history['val_accuracy'])
plt.plot(training_gru.history['val_accuracy'])
plt.title('Q3 Training & Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['RNN - Train', 'LSTM - Train', 'GRU - Train',
            'RNN - Test', 'LSTM - Test', 'GRU - Test'], loc='upper left')
plt.show()

plt.plot(training_rnn.history['loss'])
plt.plot(training_lstm.history['loss'])
plt.plot(training_gru.history['loss'])
plt.plot(training_rnn.history['val_loss'])
plt.plot(training_lstm.history['val_loss'])
plt.plot(training_gru.history['val_loss'])
plt.title('Q3 Training & Testing Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['RNN - Train', 'LSTM - Train', 'GRU - Train',
            'RNN - Test', 'LSTM - Test', 'GRU - Test'], loc='upper left')
plt.show()

# TESTING
score_rnn = model_rnn.evaluate(Test, LTest, batch_size=batchSize)
score_lstm = model_lstm.evaluate(Test, LTest, batch_size=batchSize)
score_gru = model_gru.evaluate(Test, LTest, batch_size=batchSize)

model_rnn.summary()
model_lstm.summary()
model_gru.summary()

print("RNN")
print("Test Loss: %f" % score_rnn[0])
print("Test Accuracy: %f" % score_rnn[1])
print("\nLSTM")
print("Test Loss: %f" % score_lstm[0])
print("Test Accuracy: %f" % score_lstm[1])
print("\nGRU")
print("Test Loss: %f" % score_gru[0])
print("Test Accuracy: %f" % score_gru[1])

