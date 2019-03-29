import tensorflow as tf
import numpy as np


def model_builder(embedding_, context_):
    num_units = 150

    def model(features, labels, mode, params):
        embedding = tf.get_variable(
            "embedding", shape=embedding_.shape, trainable=False, dtype=tf.float32
        )
        all_context = tf.get_variable(
            "all_context", shape=context_.shape, trainable=False, dtype=tf.int32
        )
        q_fw = tf.contrib.rnn.GRUBlockCellV2(num_units=num_units, name="q_fw")
        q_bw = tf.contrib.rnn.GRUBlockCellV2(num_units=num_units, name="q_bw")
        c_fw = tf.contrib.rnn.GRUBlockCellV2(num_units=num_units, name="c_fw")
        c_bw = tf.contrib.rnn.GRUBlockCellV2(num_units=num_units, name="c_bw")

        def init_fn(scaffold, sess):
            sess.run(embedding.initializer, {embedding.initial_value: embedding_})
            sess.run(all_context.initializer, {all_context.initial_value: context_})
            tf.logging.info("embedding initialized")

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(f"  name = {name}, shape = {features[name].shape}")

        context_id = features["context_id"]
        q = features["q"]
        batch_size = tf.shape(q)[0]
        sample_size = tf.shape(context_id)[1]
        q = tf.nn.embedding_lookup(embedding, q)
        context = tf.nn.embedding_lookup(all_context, context_id)
        context = tf.nn.embedding_lookup(embedding, context_id)
        with tf.variable_scope("q_birnn"):
            q = tf.nn.bidirectional_dynamic_rnn(q_fw, q_bw, q, dtype=tf.float32)[0]
            q = tf.concat(q, 2)
            q = q[:, -1, :]
        with tf.variable_scope("c_birnn"):
            context = tf.reshape(
                context, [batch_size * sample_size, -1, embedding_.shape[-1]]
            )
            context = tf.nn.bidirectional_dynamic_rnn(
                c_fw, c_bw, context, dtype=tf.float32
            )[0]
            context = tf.concat(context, 2)
            context = tf.reshape(context, [batch_size, sample_size, -1, 300])
            context = context[:, :, -1, :]
        q = tf.expand_dims(q, -2)
        logits = tf.matmul(context, q, transpose_b=True)
        logits = tf.squeeze(logits, -1)

        labels = tf.zeros(shape=(batch_size), dtype=tf.int32)
        one_hot = tf.one_hot(labels, depth=sample_size)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_sum(one_hot * log_probs, axis=-1)
        loss = tf.reduce_mean(loss)
        if mode == tf.estimator.ModeKeys.TRAIN:
            scaffold = tf.train.Scaffold(init_fn=init_fn)
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
            "q": tf.FixedLenFeature([20], tf.int64),
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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    emb = np.load("embedding.npy")
    print(emb.size)
    contexts = np.load("all_context.npy")
    model_fn = model_builder(emb, contexts)
    run_config = tf.estimator.RunConfig(
        model_dir="tmp",
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=2,
        log_step_count_steps=100,
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
            input_file="train.tfrecord",
            is_training=True,
            batch_size=20,
            sample_size=5,
            total_context=len(contexts),
        ),
        max_steps=200_000,
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run()
