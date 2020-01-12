import tensorflow as tf

import common


class B2:
    def __init__(self):
        X_train_uncropped, self.y_train = common.load_images_and_labels("cartoon_set", "face_shape")
        X_test_uncropped, self.y_test = common.load_images_and_labels("cartoon_set_test", "face_shape")

        self.X_train = tf.image.crop_to_bounding_box(X_train_uncropped, 150, 150, 250, 200)
        self.X_test = tf.image.crop_to_bounding_box(X_test_uncropped, 150, 150, 250, 200)

        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(250, 200, 3)
            )
        )
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(2, activation="softmax"))

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self):
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            validation_data=(self.X_test, self.y_test),
        )
        return history.history["accuracy"][-1]

    def test(self):
        _, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        return test_accuracy
