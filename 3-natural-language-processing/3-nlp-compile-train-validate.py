# Character prediction
# At end of epoch state needs to be resetted
class ResetStatesCallback(keras.callbacks.Callback):
		def on_epoch_begin(self, epoch, logs):
				self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])

# Sentiment analysis
model.compile(loss="binary_crossentropy", optimizer="adam",
							metrics=["accuracy"])
history = model.fit(train_set, epochs=5)