import tensorflow as tf

import common


class B2:
    def __init__(self):
        """Initializes the CNN-based solver for task B1.
        """
        # import train and test data
        X_train_uncropped, self.y_train = common.load_images_and_labels("cartoon_set", "eye_color")
        X_test_uncropped, self.y_test = common.load_images_and_labels("cartoon_set_test", "eye_color")

        # crop the data to include the face only
        self.X_train = tf.image.crop_to_bounding_box(X_train_uncropped, 250, 180, 30, 140)
        self.X_test = tf.image.crop_to_bounding_box(X_test_uncropped, 250, 180, 30, 140)

        # create CNN
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(30, 140, 3)
            )
        )
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(5, activation="softmax"))

        # compile CNN
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self):
        """Trains the CNN and return accuracy.

        Returns:
            float: Training accuracy.
        """
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            validation_data=(self.X_test, self.y_test),
        )
        return history.history["accuracy"][-1]

    def test(self):
        """Tests the CNN on the testing data and returns accuracy.

        Returns:
            float: Test accuracy.
        """
        _, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        return test_accuracy
