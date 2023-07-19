#!/usr/bin/env python
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from models import ResNet43_8s, UNet43_8s

import ipdb


class Affordance:
    """attention model implemented as an hourglass FCN.
    In the normal Transporter model, this component only uses one rotation,
    so the label is just sized at (320,160,1).
    """

    def __init__(self, input_shape, preprocess, learning_rate=1e-4):
        self.preprocess = preprocess

        input_shape = tuple(input_shape)

        in0, out0, global_feat = UNet43_8s(input_shape, 256, prefix='s0_d1_')
        self.conv_seq = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu',
                                   input_shape=(input_shape[0], input_shape[1], 256 + 512)),
            tf.keras.layers.Conv2D(filters=1, kernel_size=1, input_shape=(input_shape[0], input_shape[1], 256)),
        ])
        self.model = tf.keras.models.Model(inputs=[in0], outputs=[out0, global_feat])

        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metric = tf.keras.metrics.Mean(name='attention_loss')

    def forward(self, in_img):
        """Forward pass.

        in_img.shape: (160, 160, 4)
        input_data.shape: (160, 160, 4), then (None, 160, 160, 4)
        """
        input_data = self.preprocess(in_img)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        feat, global_feat = self.model([in_tensor])
        global_feat = tf.tile(global_feat, [1, feat.shape[1], feat.shape[2], 1])
        all_feat = tf.concat([feat, global_feat], axis=-1)
        logits = self.conv_seq(all_feat)

        return logits

    def forward_batch(self, in_img_batch):
        """Forward pass.

        in_img.shape: (160, 160, 4)
        input_data.shape: (160, 160, 4), then (None, 320, 320, 6)
        """
        batch_len = len(in_img_batch)
        in_tensor_batch = []
        for i in range(batch_len):
            input_data = self.preprocess(in_img_batch[i])  # (160,160,4)
            input_shape = (1,) + input_data.shape
            input_data = input_data.reshape(input_shape)  # (1,160,160,4)
            in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,160,160,4)
            in_tensor_batch.append(in_tensor)
        in_tensor = tf.concat(in_tensor_batch, axis=0)  # (batch,160,160,4)

        feat, global_feat = self.model([in_tensor])
        global_feat = tf.tile(global_feat, [1, feat.shape[1], feat.shape[2], 1])
        all_feat = tf.concat([feat, global_feat], axis=-1)
        logits = self.conv_seq(all_feat)

        return logits

    def load(self, path):
        self.model.load_weights(path)

    def save(self, filename):
        self.model.save(filename)
