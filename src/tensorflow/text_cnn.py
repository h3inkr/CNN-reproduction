import tensorflow as tf
import numpy as np


class TextCNN(tf.keras.Model):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling, and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        super(TextCNN, self).__init__()

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        
        # Convolution and max-pooling layers
        self.conv_layers = []
        for filter_size in filter_sizes:
            self.conv_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(num_filters, (filter_size, embedding_size), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 1))
                ])
            )
        
        # Fully connected output layer
        self.dense = tf.keras.layers.Dense(num_classes)
        
        # Ensure that dropout rate is valid (between 0 and 1)
        dropout_rate = max(0.0, min(1.0 - l2_reg_lambda, 0.5))  # Ensure it does not exceed 0.5
        print(f"Dropout rate: {dropout_rate}")  # Debugging line
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)  # Embedding layer
        
        pooled_outputs = []
        for conv_layer in self.conv_layers:
            x_conv = conv_layer(x)
            # Flatten the pooled output (preserving batch size)
            pooled_outputs.append(tf.reshape(x_conv, [x_conv.shape[0], -1]))  # Flatten all dimensions except batch size
        
        # Concatenate all pooled features
        x = tf.concat(pooled_outputs, axis=-1)  # Concatenate along channels (filters)

        # Dropout
        x = self.dropout(x, training=training)

        # Final output layer
        x = self.dense(x)
        return x
