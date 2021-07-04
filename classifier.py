import tensorflow as tf
import sys

from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

IMG_WIDTH = 28
IMG_HEIGHT = 28
EPOCHS = 65


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )
    x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2
    )

    model = get_model()

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(cooldown=5, verbose=1)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=EPOCHS,
        validation_data=(x_valid, y_valid),
        callbacks=[lr_callback, stop_early],
    )

    model.evaluate(x_test, y_test)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")


def get_model():

    data_aug = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomTranslation(
                0.1,
                0.1,
                input_shape=(IMG_WIDTH, IMG_HEIGHT, 1),
            ),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.05),
        ]
    )

    model = tf.keras.Sequential(
        [
            data_aug,
            # conv layer 0
            Conv2D(32, 5, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 1
            Conv2D(64, 5, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 2
            Conv2D(32, 5, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 3
            Conv2D(16, 7, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 4
            Conv2D(8, 9, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 5
            Conv2D(32, 7, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 6
            Conv2D(32, 5, activation="relu", padding="same"),
            BatchNormalization(),
            # conv layer 7
            Conv2D(64, 3, activation="relu", padding="same"),
            BatchNormalization(),
            tf.keras.layers.Flatten(),
            # dense layer 0
            Dense(units=352, activation="relu"),
            BatchNormalization(),
            # dense layer 1
            Dense(units=448, activation="relu"),
            BatchNormalization(),
            # dense layer 2
            Dense(units=160, activation="relu"),
            BatchNormalization(),
            # dense layer 3
            Dense(units=256, activation="relu"),
            BatchNormalization(),
            # dense layer 3
            Dense(units=160, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    main()
