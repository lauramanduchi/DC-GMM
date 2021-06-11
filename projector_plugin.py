import os
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ProjectorPlugin:
    SPRITE_FILENAME = "sprites.png"
    METADATA_FILE = "metadata.tsv"
    EMBEDDING_TENSOR_NAME = "embedding"

    def __init__(self, log_dir_fq_path, x, step=0):
        self.log_dir_fq_path = log_dir_fq_path
        self.config = projector.ProjectorConfig()
        self.embedding = self.config.embeddings.add()
        self.embedding.tensor_name = ProjectorPlugin.EMBEDDING_TENSOR_NAME

        tensor_embeddings = tf.Variable(x, name=ProjectorPlugin.EMBEDDING_TENSOR_NAME)
        saver = tf.compat.v1.train.Saver([tensor_embeddings])
        saver.save(sess=None, global_step=step,
                   save_path=self.log_dir_fq_path + '/' + ProjectorPlugin.EMBEDDING_TENSOR_NAME + '.ckpt')

    def save_labels(self, y):
        with open(os.path.join(self.log_dir_fq_path, ProjectorPlugin.METADATA_FILE), 'w') as f:
            for label in y:
                f.write('{}\n'.format(label))

        self.embedding.metadata_path = ProjectorPlugin.METADATA_FILE

    def save_image_sprites(self, image_array, img_width, img_height, num_channels, invert=False):
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
            plt.imsave(os.path.join(self.log_dir_fq_path, ProjectorPlugin.SPRITE_FILENAME), sprite_image, cmap='gray')
        else:
            sprite_image *= 255.
            cv2.imwrite(os.path.join(self.log_dir_fq_path, ProjectorPlugin.SPRITE_FILENAME), sprite_image)

        self.embedding.sprite.image_path = ProjectorPlugin.SPRITE_FILENAME
        self.embedding.sprite.single_image_dim.extend([img_width, img_height])

    def finalize(self):
        projector.visualize_embeddings(self.log_dir_fq_path, self.config)
