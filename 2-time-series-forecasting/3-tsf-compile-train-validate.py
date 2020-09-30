# Compile, train and validate model

def last_time_step_mse(Y_true, Y_pred):
		return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])

history = model.fit(X_train, Y_train[:, 3::2], epochs=20,
										validation_data=(X_valid, Y_valid[:, 3::2]))