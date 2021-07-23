import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs


def build_classifier_model() -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation="relu", name='classifier')(net)

    return tf.keras.Model(text_input, net)


if __name__ == "__main__":
    EPOCH: int = 100

    print("Loading dataset")
    raw_data: pd.DataFrame = pd.read_csv("./dataset.csv")

    # Split train and test
    x_train, x_test, y_train, y_test = train_test_split(raw_data["comment"], raw_data["useful"], test_size=0.2,
                                                        shuffle=True)

    # Load BERT
    tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
    tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

    # Layer to preprocess input text, convert to tokens
    bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
    bert_model = hub.KerasLayer(tfhub_handle_encoder)

    model: tf.keras.Model = build_classifier_model()

    # Loss function
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    model.compile(optimizer=optimizers.Adam(1e-5), loss=loss, metrics=metrics)

    print(f'Training model')
    history = model.fit(x=x_train, y=y_train, epochs=EPOCH)

    loss, accuracy = model.evaluate(x=x_test, y=y_test)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    from datetime import datetime

    tmp_file: str = './models/bert-' + str(datetime.now().strftime("%Y%m%d%H%M"))
    model.save(tmp_file, include_optimizer=False, save_format="tf")
    try:
        tfjs.converters.save_keras_model(model, "./tfjs_model")
    except Exception as e:
        print(e)

    # Test
    samples: list = ["Item reached",
                     "Delivery time slow",
                     "This product works as intended",
                     "The bass was a little too loud, but overall the speaker works fine",
                     "Terbaikterbaikterbaikterbaikterbaikterbaikterbaik Terbaikterbaikterbaikterbaikterbaikterbaikterbaik Terbaikterbaikterbaikterbaikterbaikterbaikterbaik Terbaikterbaikterbaikterbaikterbaikterbaikterbaik Terbaikterbaikterbaikterbaikterbaikterbaikterbaik Terbaikterbaikterbaikterbaikterbaikterbaikterbaik",
                     ]

    r = tf.sigmoid(model(tf.constant(samples)))
    print(r)
