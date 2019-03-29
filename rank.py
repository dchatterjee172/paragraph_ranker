import tensorflow as tf
import numpy as np


def model_builder(embedding_, context_):
    def model(features, labels, mode, params):
        embedding = tf.get_variable("embedding", size=embedding_.size, trainable=False)
        context = tf.get_variable("embedding", size=context_.size, trainable=False)

        def init_fn(scaffold, sess):
            sess.run(embedding.initializer, {embedding.initial_value: embedding_})
            sess.run(context.initializer, {context.initial_value: context_})

        context_id = features["context_id"]
        q = features["q"]
        if mode == tf.estimator.ModeKeys.TRAIN:
            scaffold = tf.train.Scaffold(init_fn=init_fn)
            loss = 0
            optimizer = tf.train.AdamOptimizer()
            train = optimizer.minimize(
                loss, global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train, scaffold=scaffold
            )
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    return model


def input_fn(input_file, is_training, batch_size, sample_size, total_context):
    name_to_features = {
        "context_id": tf.FixedLenFeature([1], tf.int64),
        "q": tf.FixedLenFeature([300], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        sample = tf.random.uniform(shape=[sample_size - 1], max_val=total_context)
        example["context_id"] = tf.concat([example["context_id"], sample])
        return example

    d = tf.data.TFRecordDataset(input_file)
    drop_remainder = False
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)
        drop_remainder = True
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder,
        )
    )
    return d


if __name__ == "__main__":
    emb = np.load("embedding.npy")
    contexts = np.load("all_context.npy")
