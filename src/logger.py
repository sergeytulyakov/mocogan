"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        with self.writer.as_default():
            tf.summary.image(name=tag, data=images, step=step)
            self.writer.flush()

    def video_summary(self, tag, videos, step):
        with self.writer.as_default():
            sh = list(videos.shape)
            sh[-1] = 1

            separator = np.zeros(sh, dtype=videos.dtype)
            videos = np.concatenate([videos, separator], axis=-1)

            for i, vid in enumerate(videos):
                # Concat a video
                v = vid.transpose(1, 2, 3, 0)
                v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]
                tf.summary.image(name='%s/%d' % (tag, i), data=v, step=step)
            self.writer.flush()
