import common


class A1:
    def __init__(self):
        X_train, y_train = common.load_images_and_labels("celeba", "gender")
        X_test, y_test = common.load_images_and_labels("celeba_test", "gender")

        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(218, 178, 3)
            )
        )
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(2, activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self):
        model.fit(
            X_train, y_train, epochs=5, validation_data=(X_test, y_test),
        )

    def test(self):
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
        return test_accuracy
