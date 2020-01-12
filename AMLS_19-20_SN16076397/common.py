import os

import cv2
import dlib
import numpy as np
import tensorflow as tf

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def load_images_and_labels(images_dirname, csv_label):
    """Loads images and labels.
    
    Args:
        images_dirname (str): Name of the directory which holds the data.
        csv_label (str): Name of the column from labels.csv to output.

    Returns:
        images (np.ndarray): 4-d array of images, in format np.float32, values between 0 and 1.
        labels (np.ndarray): 1-d array of labels, in format np.int8.
    """
    # get paths of images
    images_dir = os.path.join("./Datasets", images_dirname, "img")
    image_paths = sorted(
        [os.path.join(images_dir, l) for l in os.listdir(images_dir)],
        key=lambda x: int(x.split(".")[1].split("/")[-1]),
    )

    images = []
    for image_path in image_paths:
        image = (
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(image_path), dtype=np.float32
            )
            / 255
        )
        images.append(image)
    images = np.array(images)

    # get labels
    labels_file = open(os.path.join("./Datasets", images_dirname, "labels.csv"), "r")
    lines = labels_file.readlines()
    labels_file.close()
    column = lines[0].split().index(csv_label) + 1
    labels = []
    for line in lines[1:]:
        line = line.split()
        value_to_append = int(line[column])
        # if -1, convert to 0
        labels.append(0 if value_to_append == -1 else value_to_append)
    labels = np.array(labels, dtype=np.int8)

    return images, labels


def shape_to_np(shape):
    """Converts shape returned by dlib into numpy array.

    Args:
        shape (dlib.full_object_detection): Object returned by dlib.
    Returns:
        np.ndarray: Coordinates of the landmarks.
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype="int")

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    """Converts bounding predicted by dlib into (x, y, w, h) format.
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def run_dlib_shape(image):
    """Detects landmarks.

    The function detects the landmarks of the face, then returns the
    landmarks and resized image.
    """
    resized_image = image.astype("uint8")

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype("uint8")

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def images_into_landmarks(images, labels):
    """Detects landmarks on all images, returns only where landmarks could be found.
    
    Args:
        images (np.ndarray): 4-d array of images, in format np.float32, values between 0 and 1.
        labels (np.ndarray): 1-d array of labels, in format np.int8.
    Returns:
        landmarks (np.ndarray): 3-d array of images, in format np.int16.
        labels (np.ndarray): 1-d array of labels, in format np.int8.
        undetectable (np.ndarray): 1-d array of indexes at which face couldn't be detected. In format int.
    """
    landmarks_detected = []
    labels_detected = []
    undetectable = []
    for i, image in enumerate(images):
        dlibout, _ = run_dlib_shape(image * 255.0)
        if dlibout is not None:
            landmarks_detected.append(dlibout)
            labels_detected.append(labels[i])
        else:
            undetectable.append(i)
    return (
        np.array(landmarks_detected, dtype=np.int16),
        np.array(labels_detected, dtype=np.int8),
        undetectable,
    )


def flatten_examples(examples):
    """Flattens ndarray except the top dimension.

    Args:
        examples (np.ndarray): Array of examples, each can contain more arrays.
    Returns:
        np.ndarray: 2-d array, where each element is a flattened example.
    """
    return examples.reshape((examples.shape[0], -1))
