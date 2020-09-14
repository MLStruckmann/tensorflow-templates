import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

### TF Admin
tf.__version__
tf.config.list_physical_devices()

### Building a model: Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
# or
model = keras.models.Sequential([
    keras.layers.Dense(30, activation = "relu"),
    keras.layers.Dense(1)
])

### Building a model: Functional API
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

### Building a model: Subclassing API
class WideAndDeepModel(keras.Model):

    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs) # handles standard args (e.g., name)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)

    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()

### Compiling and training a model
model.compile(loss = "mean_squared_error",
              optimizer = "sgd") # you can also pass optimizer objects, see below

history = model.fit(X_train, y_train,
                    epochs = 20,
                    validation_data = (X_valid, y_valid)
                    #callbacks = [checkpoint_callback, early_stopping_callback, chris_schmitz_callback, tb_cb] # see below!
)

### Getting predictions
model.predict([input1, input2, input3])

### Saving and loading a model
model.save("my_name.h5")
model = keras.models.load_model("my_name.h5")

### Using Callbacks

# This one saves it and overwrites the old one every epoch, provided the validation score is Banging Harder
checkpoint_callback = keras.callbacks.ModelCheckpoint("my_name.h5",save_best_only=True)

# This one just stops training if the loss metric hasn't improved in 10 epochs and rolls back to the best model
early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights= True)

# And here's a custom one. The functions need exactly these names, and there are many more
# For the parameters of each function: I'm not sure if these are exactly right.
class ChrisSchmitzCallback(keras.callbacks.Callback):
    def on_train_begin(self, epoch, logs):
        print('hello it is callback time')

    def on_epoch_end(self, epoch, logs):
        print('another epoch over. time flies')

    #on_batch_begin/end, on_test_begin/end, on_test_batch_begin/end, on_predict_batch_begin/end, etc...

chris_schmitz_callback = ChrisSchmitzCallback()

### Using TensorBoard

# Setting run log directory and getting one for the run

import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# Logging part of the run in Tensorboard with a callback
tb_cb = keras.callbacks.TensorBoard(run_logdir)

# Viewing the tensorboard
# tensorboard --logdir=./my_logs

### Training Hacks: Vanishing & Exploding Gradients

## Kernel Initializer
# In a dense layer. For He initialization, ReLU has to be the activation fn
# For SELU activation fn, use LeCun initialization
layerino = keras.layers.Dense(200,activation='ReLU',kernel_initializer='he_normal')

## Different activation functions
# Leaky ReLU needs a new layer, and no activation in the dense one:
layerino = keras.layers.Dense(200)
layer_two = keras.layers.LeakyReLU(alpha=0.2)[layerino]

# SELU doesn't need a new layer. But check the notes for when to use that (it's rare)

## Batch normalization
# More or less standard after every layer
# You can pass momentum = [something close to 1]
# And you can put them before the activation - by leaving the act. out of the layer and putting it after, as above
keras.layers.BatchNormalization()[layer_two]

## Gradient Clipping
optimizer = keras.optimizers.SGD(clipvalue = 1.0)
optimizer = keras.optimizers.SGD(clipnorm = 1.0)

### Training Hacks: Faster Optimizers
optimizer = keras.optimizers.SGD(momentum = 0.9)
optimizer = keras.optimizers.SGD(momentum = 0.9, nesterov= True)

# or just a different one
optimizer = keras.optimizers.Adamax() #etc

### Preventing Overfitting

# Kernel regularization
# this is also introducing functools
from functools import partial

RegularizedDense = partial(keras.layers.Dense,
                        activation="elu",
                        kernel_initializer="he_normal",
                        kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
                        keras.layers.Flatten(input_shape=[28, 28]),
                        RegularizedDense(300),
                        RegularizedDense(100),
                        RegularizedDense(10, activation="softmax",
                        kernel_initializer="glorot_uniform")
])
# Dropout
# usually 20-50% and in the last 1-3 layers before the output
# if it's shitcanning performance, try only put it after the last hidden layer
keras.layers.Dropout(rate = 0.2)

# Maxnorm Regularization
normedlayer = keras.layers.Dense(300, activation = 'relu', kernel_constraint=keras.constraints.max_norm(1.0))

### Transfer Learning
# Loading a model and picking off layers to reuse
model_a = keras.models.load_model('model_a.h5')
reused_layers = keras.models.Sequential(model_a.layers[:-1])

# Adding new output layer
reused_layers.add(keras.layers.Dense(1, activation = "sigmoid"))

# Cloning the model. For some reason
model_b = keras.models.clone_model(reused_layers)
model_b.set_weights(reused_layers.get_weights())

# Freezing all the reused layers for a few epochs
# You have to compile after every freeze and unfreeze!
for layer in model_b.layers[:-1]:
    layer.trainable = False

model_b.compile(loss = 'binary_crossentropy',
                optimizer = 'sgd',
                metrics= ['accuracy'])

# Then train the model for a few epochs, unfreeze (some or all) reused layers,
# reduce the learning rate, continue the training, etc etc
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-2

model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))

### Good Defaults
# A default model and some data to test it

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full/255

nodes_per_hidden = 300
out_classes = 10

input_ = keras.layers.Input(shape=X_train_full.shape[1:])
flat = keras.layers.Flatten()(input_)
batch0 = keras.layers.BatchNormalization()(flat)
hidden1 = keras.layers.Dense(units = nodes_per_hidden,activation= "elu", kernel_initializer='he_normal')(batch0)
batch1 = keras.layers.BatchNormalization()(hidden1)
hidden2 = keras.layers.Dense(nodes_per_hidden, "elu", kernel_initializer='he_normal')(batch1)
batch2 = keras.layers.BatchNormalization()(hidden2)
drop1 = keras.layers.Dropout(rate = 0.2)(batch2)
#etc if needed
out = keras.layers.Dense(out_classes, activation="softmax")(drop1)

model = keras.Model(inputs=[input_], outputs=[out])

optimizer = keras.optimizers.Nadam()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

history = model.fit(X_train_full, y_train_full,
                    epochs = 30,
                    callbacks = [keras.callbacks.EarlyStopping(patience=10)],
                    validation_split=0.3)


