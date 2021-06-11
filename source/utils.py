import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import socket
import logging
import random
import cv2
from PIL import Image
from matplotlib import cm

from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

from source.constants import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


def setup_logger(results_path, create_stdlog):
    """Setup a general logger which saves all logs in the experiment folder"""

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(results_path))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    if create_stdlog:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def get_assigned_cluster_mapping(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_assignment(w.max() - w)

    return ind[:, 1]


def make_confusion_matrix(y_true, y_pred, num_classes):
    assert len(y_pred) == len(y_true), "Lengths must match"
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int)

    cluster_mapping = list(get_assigned_cluster_mapping(y_true, y_pred))

    for i in range(len(y_pred)):
        conf_mat[y_true[i]][cluster_mapping[y_pred[i]]] += 1

    return conf_mat


def plot_image_rectangle(image_array, img_width, img_height, num_channels, path=None):
    num_width = image_array.shape[1]
    num_height = image_array.shape[0]

    sprite_image = np.ones((img_height * num_height, img_width * num_width, num_channels))

    for i in range(num_height):
        for j in range(num_width):
            this_img = image_array[i, j]
            sprite_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width, 0:num_channels] \
                = this_img

    if path is not None:
        sprite_image *= 255.
        im = Image.fromarray(bgr_to_rgb(np.uint8(sprite_image)))
        im.save(path)
    else:
        cv2.imshow("image", sprite_image)
        cv2.waitKey(0)


def plot_image_square(image_array, img_width, img_height, num_channels, path=None, invert=False):
    num_images = image_array.shape[0]

    # Ensure shape
    if num_channels == 1:
        image_array = np.reshape(image_array, (-1, img_width, img_height))
    else:
        image_array = np.reshape(image_array, (-1, img_width, img_height, num_channels))

    # Invert pixel values
    if invert:
        image_array = 1 - image_array

    image_array = np.array(image_array)

    # Plot images in square
    n_plots = int(np.ceil(np.sqrt(num_images)))

    # Save image
    if num_channels == 1:
        sprite_image = np.ones((img_height * n_plots, img_width * n_plots))
    else:
        sprite_image = np.ones((img_height * n_plots, img_width * n_plots, num_channels))

    # fill the sprite templates
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < image_array.shape[0]:
                this_img = image_array[this_filter]
                if num_channels == 1:
                    sprite_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = this_img
                else:
                    sprite_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width,
                    0:num_channels] = this_img

    # save the sprite image
    if num_channels == 1:
        if path is not None:
            plt.imsave(path, sprite_image, cmap='gray')
        else:
            plt.axis('off')
            plt.imshow(sprite_image, cmap='gray')
            plt.show()
            plt.close()
    else:
        if path is not None:
            sprite_image *= 255.
            cv2.imwrite(path, sprite_image)
        else:
            cv2.imshow("image", sprite_image)
            cv2.waitKey(0)


def bgr_to_rgb(bgr):
    return bgr[:, :, ::-1]
