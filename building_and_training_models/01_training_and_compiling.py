### Compiling and training a model
model.compile(loss = "mean_squared_error",
              optimizer = "sgd") # you can also pass optimizer objects, see below

history = model.fit(X_train, y_train,
                    epochs = 20,
                    validation_data = (X_valid, y_valid)
                    #callbacks = [checkpoint_callback, early_stopping_callback, chris_schmitz_callback, tb_cb] # see below!
)

### Getting predictions
model.predict([input1, input2, input3])

### Saving and loading a model
model.save("my_name.h5")
model = keras.models.load_model("my_name.h5")