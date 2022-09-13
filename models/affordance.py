#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from models import ResNet43_8s, ResNet36_4s, ResNet53_8s, ResNet61_8s, UNet43_8s

import ipdb


class Affordance:
    """attention model implemented as an hourglass FCN.
    In the normal Transporter model, this component only uses one rotation,
    so the label is just sized at (320,160,1).
    """

    def __init__(self, input_shape, preprocess, unet=1, out_logits=1, learning_rate=1e-4):
        self.preprocess = preprocess
        self.out_logits = out_logits

        input_shape = tuple(input_shape)
        self.unet = unet

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        if self.unet:
            in0, out0, global_feat = UNet43_8s(input_shape, 256, prefix='s0_d1_')
            self.conv_seq = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu',
                                       input_shape=(320, 320, 256 + 512)),
                tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(320, 320, 256)),
            ])
            self.model = tf.keras.models.Model(inputs=[in0], outputs=[out0, global_feat])
        else:
            d_in, d_out = ResNet43_8s(input_shape, 1)
            self.model = tf.keras.models.Model(inputs=[d_in], outputs=[d_out])

        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metric = tf.keras.metrics.Mean(name='attention_loss')

    def forward(self, in_img, apply_softmax=True):
        """Forward pass.

        in_img.shape: (320, 160, 6)
        input_data.shape: (320, 320, 6), then (None, 320, 320, 6)
        """
        # input_data = np.pad(in_img, self.padding, mode='constant')
        input_data = self.preprocess(in_img)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        if self.unet:
            feat, global_feat = self.model([in_tensor])
            global_feat = tf.tile(global_feat, [1, 320, 320, 1])
            all_feat = tf.concat([feat, global_feat], axis=-1)
            logits = self.conv_seq(all_feat)
        else:
            logits = self.model(in_tensor)

        return logits

    def train(self, in_img, p, gt):
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            output = self.forward(in_img, apply_softmax=False)

            # Compute loss
            output = output[:, p[0], p[1], :]
            loss = tf.keras.losses.MAE(gt, output)
            # loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
            loss = tf.reduce_mean(loss)

        # Backpropagate
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(
            zip(grad, self.model.trainable_variables))

        self.metric(loss)
        return np.float32(loss)

    def load(self, path):
        self.model.load_weights(path)

    def save(self, filename):
        self.model.save(filename)
