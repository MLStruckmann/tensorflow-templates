# Conv layer for long term memory and GRU for regression

model = keras.models.Sequential([
		keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
												input_shape=[None, 1]),
		keras.layers.GRU(20, return_sequences=True),
		keras.layers.GRU(20, return_sequences=True),
		keras.layers.TimeDistributed(keras.layers.Dense(10))
])
