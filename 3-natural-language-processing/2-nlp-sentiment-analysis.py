# Sentiment anaylsis model

embed_size = 128
model = keras.models.Sequential([
		keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
													 input_shape=[None]),
		keras.layers.GRU(128, return_sequences=True),
		keras.layers.GRU(128),
		keras.layers.Dense(1, activation="sigmoid")
])