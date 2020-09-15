# Compile train and validate model

# Optimizer
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
optimizer = "adam"

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = "sparse_categorical_crossentropy"
loss = ["sparse_categorical_crossentropy", "mse"] # for localization

# Loss weights for multiple losses
loss_weights = [0.8, 0.2], # for localization, numbers depend on what you care most about

# Compile model
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=["accuracy"]
              #loss_weights=loss_weights
              )

# Model summary
model.summary()

# Train model
epochs=10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)