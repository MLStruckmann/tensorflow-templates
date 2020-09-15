# See 00 for where to pass these callbacks

# This one saves it and overwrites the old one every epoch, provided the validation score is Banging Harder
checkpoint_callback = keras.callbacks.ModelCheckpoint("my_name.h5",save_best_only=True)

# This one just stops training if the loss metric hasn't improved in 10 epochs and rolls back to the best model
early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights= True)

# And here's a custom one. The functions need exactly these names, and there are many more
# For the parameters of each function: I'm not sure if these are exactly right.
class ChrisSchmitzCallback(keras.callbacks.Callback):
    def on_train_begin(self, epoch, logs):
        print('hello it is callback time')

    def on_epoch_end(self, epoch, logs):
        print('another epoch over. time flies')

    #on_batch_begin/end, on_test_begin/end, on_test_batch_begin/end, on_predict_batch_begin/end, etc...

chris_schmitz_callback = ChrisSchmitzCallback()

### Using TensorBoard

# Setting run log directory and getting one for the run

import os
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

# Logging part of the run in Tensorboard with a callback
tb_cb = keras.callbacks.TensorBoard(run_logdir)

# Viewing the tensorboard
# tensorboard --logdir=./my_logs
