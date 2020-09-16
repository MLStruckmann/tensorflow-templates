# LSTM model

model = keras.models.Sequential([
		keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
		keras.layers.LSTM(20, return_sequences=True),
		keras.layers.TimeDistributed(keras.layers.Dense(10))
])