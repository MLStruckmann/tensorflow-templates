# GRU model

model = keras.models.Sequential([
		keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
		keras.layers.GRU(20, return_sequences=True),
		keras.layers.TimeDistributed(keras.layers.Dense(10))
])