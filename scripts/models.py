import tensorflow as tf


def time_series_model(days_to_predict,
                      neurons='64',
                      optimizer='adam',
                      single_step=True,
                      model_save_path="model",
                      loss="mae"):
    if single_step:
        days_to_predict = 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(neurons),
        tf.keras.layers.Dense(days_to_predict)
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=[loss])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                           monitor=loss,
                                           save_weights_only=True,
                                           save_best_only=True,
                                           mode='auto')
    ]

    return model, callbacks
