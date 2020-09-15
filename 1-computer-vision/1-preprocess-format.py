# Data preprocessing

'''MISSING: Train, val, test split'''

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    '''MISSING: Manual scaling of pixels'''
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# Alternative:
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_set = train_set.shuffle(1000)
# train_set = train_set.map(preprocess).cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
# valid_set = valid_set.map(preprocess).cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
# test_set = test_set.map(preprocess).cache().batch(batch_size).prefetch(buffer_size=AUTOTUNE)