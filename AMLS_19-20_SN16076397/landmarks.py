import os

import cv2
import dlib
import keras
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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


def extract_features_labels(images_dirname):
    """Extracts features for all images under specified path.

    Args:
        images_dirname (str): Folder containing the images. E.g. celeba.
    
    Returns:
        Tuple[Iterable, Iterable]: Array containing 68 landmark points
            for each image in which a face was detected; an array of
            labels found in labels.csv.
    """
    # get paths of images
    images_dir = os.path.join("./Datasets", images_dirname, "img")
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]

    # get labels, convert into dict, change -1 to 0
    labels_file = open(os.path.join("./Datasets", images_dirname, "labels.csv"), "r")
    lines = labels_file.readlines()
    labels_file.close()
    all_labels = {}
    for line in lines[1:]:
        line = line.split()
        if "celeba" in images_dirname:
            all_labels[int(line[0])] = [
                line[1],
                (int(line[2]) + 1) // 2,
                (int(line[3]) + 1) // 2,
            ]
        if "cartoon_set" in images_dirname:
            all_labels[int(line[0])] = [int(line[1]), int(line[2]), line[3]]
        else:
            raise (NotImplementedError)

    if os.path.isdir(images_dir):
        all_features = []
        labels = []

        for image_path in image_paths:
            img = keras.preprocessing.image.img_to_array(
                keras.preprocessing.image.load_img(
                    image_path, target_size=None, interpolation="bicubic"
                )
            )
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                image_id = int(image_path.split(".")[1].split("/")[-1])
                labels.append(all_labels[image_id])

    landmark_features = np.array(all_features)
    return landmark_features, labels
