import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2])
    return tf.reduce_mean((2 * intersection) / (union + 1e-5))

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true, axis=[0, 1, 2]) + tf.reduce_sum(y_pred, axis=[0, 1, 2]) - intersection
    return tf.reduce_mean((intersection + 1e-5) / (union + 1e-5))

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    return 0.5 * (1 - dice_coef(y_true, y_pred)) + 0.5 * tf.keras.losses.categorical_crossentropy(y_true, y_pred)
