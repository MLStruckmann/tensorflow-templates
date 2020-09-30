# Load pretrained embedding layers

import tensorflow_hub as hub
model = keras.Sequential([
		hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
									 dtype=tf.string, input_shape=[], output_shape=[50]),
		keras.layers.Dense(128, activation="relu"),
		keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
							metrics=["accuracy"])