### Transfer Learning
# Loading a model and picking off layers to reuse
model_a = keras.models.load_model('model_a.h5')
reused_layers = keras.models.Sequential(model_a.layers[:-1])

# Adding new output layer
reused_layers.add(keras.layers.Dense(1, activation = "sigmoid"))

# Cloning the model. For some reason
model_b = keras.models.clone_model(reused_layers)
model_b.set_weights(reused_layers.get_weights())

# Freezing all the reused layers for a few epochs
# You have to compile after every freeze and unfreeze!
for layer in model_b.layers[:-1]:
    layer.trainable = False

model_b.compile(loss = 'binary_crossentropy',
                optimizer = 'sgd',
                metrics= ['accuracy'])

# Then train the model for a few epochs, unfreeze (some or all) reused layers,
# reduce the learning rate, continue the training, etc etc
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-2

model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))