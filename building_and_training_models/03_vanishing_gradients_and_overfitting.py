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