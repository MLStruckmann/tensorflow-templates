# A default model and some data to test it

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full = X_train_full/255

nodes_per_hidden = 300
out_classes = 10

input_ = keras.layers.Input(shape=X_train_full.shape[1:])
flat = keras.layers.Flatten()(input_)
batch0 = keras.layers.BatchNormalization()(flat)
hidden1 = keras.layers.Dense(units = nodes_per_hidden,activation= "elu", kernel_initializer='he_normal')(batch0)
batch1 = keras.layers.BatchNormalization()(hidden1)
hidden2 = keras.layers.Dense(nodes_per_hidden, "elu", kernel_initializer='he_normal')(batch1)
batch2 = keras.layers.BatchNormalization()(hidden2)
drop1 = keras.layers.Dropout(rate = 0.2)(batch2)
#etc if needed
out = keras.layers.Dense(out_classes, activation="softmax")(drop1)

model = keras.Model(inputs=[input_], outputs=[out])

optimizer = keras.optimizers.Nadam()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

history = model.fit(X_train_full, y_train_full,
                    epochs = 30,
                    callbacks = [keras.callbacks.EarlyStopping(patience=10)],
                    validation_split=0.3)
