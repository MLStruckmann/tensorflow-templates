# Simple character prediction model (stateful)

model = keras.models.Sequential([
		keras.layers.GRU(128, return_sequences=True, stateful=True,
										 dropout=0.2, recurrent_dropout=0.2,
										 batch_input_shape=[batch_size, None, max_id]),
		keras.layers.GRU(128, return_sequences=True, stateful=True,
										 dropout=0.2, recurrent_dropout=0.2),
		keras.layers.TimeDistributed(keras.layers.Dense(max_id,
																										activation="softmax"))
])