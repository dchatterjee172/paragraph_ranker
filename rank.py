import tensorflow as tf
import numpy as np


def model_builder(embedding_, context_):
    def model(features, labels, mode, params):
        embedding = tf.get_variable("embedding", shape=embedding_.size, trainable=False)
        context = tf.get_variable("context", shape=context_.size, trainable=False)

        def init_fn(scaffold, sess):
            sess.run(embedding.initializer, {embedding.initial_value: embedding_})
            sess.run(context.initializer, {context.initial_value: context_})

        context_id = features["context_id"]
        q = features["q"]
        if mode == tf.estimator.ModeKeys.TRAIN:
            scaffold = tf.train.Scaffold(init_fn=init_fn)
            loss = tf.constant(0.0)
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


def input_fn_builder(input_file, is_training, batch_size, sample_size, total_context):
    def input_fn():
        name_to_features = {
            "context_id": tf.FixedLenFeature([1], tf.int64),
            "q": tf.FixedLenFeature([300], tf.int64),
        }

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            sample = tf.random.uniform(
                shape=[sample_size - 1], maxval=total_context, dtype=tf.int64
            )
            example["context_id"] = tf.concat([example["context_id"], sample], axis=-1)
            return example

        d = tf.data.TFRecordDataset(input_file)
        drop_remainder = False
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            drop_remainder = True
        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )
        return d

    return input_fn


if __name__ == "__main__":
    emb = np.load("embedding.npy")
    contexts = np.load("all_context.npy")
    model_fn = model_builder(emb, contexts)
    run_config = tf.estimator.RunConfig(
        model_dir="tmp",
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=2,
        log_step_count_steps=1,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_builder(
            input_file="test.tfrecord",
            is_training=False,
            batch_size=2,
            sample_size=5,
            total_context=len(contexts),
        ),
        steps=100,
        start_delay_secs=0,
        throttle_secs=120,
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_builder(
            input_file="test.tfrecord",
            is_training=False,
            batch_size=2,
            sample_size=5,
            total_context=len(contexts),
        ),
        max_steps=200,
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
