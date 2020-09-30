# Transfer learning with pretrained model

base_model = keras.applications.xception.Xception(weights="imagenet",include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers: # Freeze base layers
    layer.trainable = False

# Compile, train & validate

for layer in base_model.layers: # Unfreeze base layers
    layer.trainable = True

# Compile, train & validate