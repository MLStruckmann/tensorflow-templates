# Simple time series forecasting models

# Sequence to vector 1
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
	keras.layers.SimpleRNN(1)
])

# Sequence to vector 2
model = keras.models.Sequential([
		keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
		keras.layers.SimpleRNN(20),
		keras.layers.Dense(1) # --> interchangeable activ func
])

# Sequence to sequence
model = keras.models.Sequential([
		keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
		keras.layers.SimpleRNN(20, return_sequences=True),
		keras.layers.TimeDistributed(keras.layers.Dense(10)) # Could also be Dense(10)
])