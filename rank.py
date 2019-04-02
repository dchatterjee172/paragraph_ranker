import tensorflow as tf
import numpy as np


def model_builder(embedding_, context_, sample_size):
    num_units = 50

    def model(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
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

        context = features["context_id"]
        q = features["q"]
        batch_size = tf.shape(q)[0]
        q = tf.nn.embedding_lookup(embedding, q)
        context = tf.nn.embedding_lookup(all_context, context)
        context = tf.nn.embedding_lookup(embedding, context)
        with tf.variable_scope("q_birnn", initializer=tf.glorot_uniform_initializer):
            q = tf.nn.bidirectional_dynamic_rnn(q_fw, q_bw, q, dtype=tf.float32)[0]
        q = tf.concat((q[1][:, 0, :], q[0][:, -1, :]), axis=-1)
        with tf.variable_scope("c_birnn", initializer=tf.glorot_uniform_initializer):
            context = tf.reshape(
                context, [batch_size * sample_size, -1, embedding_.shape[-1]]
            )
            context = tf.layers.separable_conv1d(
                context, 50, 5, padding="same", activation=tf.nn.leaky_relu, strides=2
            )
            context = tf.layers.batch_normalization(context, training=is_training)
            context = tf.nn.bidirectional_dynamic_rnn(
                c_fw, c_bw, context, dtype=tf.float32
            )[0]
        context = tf.concat([context[1][:, 0, :], context[0][:, -1, :]], axis=-1)
        context = tf.reshape(context, [batch_size, sample_size, num_units * 2])
        q = tf.expand_dims(q, -2)
        q = tf.layers.dropout(q, 0.2, training=is_training)
        context = tf.layers.dropout(context, 0.2, training=is_training)
        logits = tf.matmul(context, q, transpose_b=True) / tf.sqrt(
            tf.constant(num_units * 2.0)
        )
        logits = tf.squeeze(logits, -1)
        labels = tf.zeros(shape=(batch_size), dtype=tf.int32)
        labels_one_hot = tf.one_hot(labels, depth=sample_size)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_one_hot, logits=logits
        )
        loss = loss[:, 0] + tf.reduce_mean(loss[:, 1:], axis=-1)
        loss = tf.reduce_mean(loss)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        tp = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        if mode == tf.estimator.ModeKeys.TRAIN:
            scaffold = tf.train.Scaffold(init_fn=init_fn)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            var = tf.trainable_variables()
            grads = tf.gradients(loss, var)
            clipped_grad, norm = tf.clip_by_global_norm(grads, 0.5)
            tf.summary.scalar("grad_norm", norm)
            tf.summary.scalar("tp", tp)
            tf.summary.histogram("predictions", predictions)
            tf.summary.histogram("logits", logits)
            tf.summary.histogram("labels", tf.argmax(labels_one_hot, -1))
            for v in tf.trainable_variables():
                tf.summary.histogram(v.name, v)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train = optimizer.apply_gradients(
                zip(clipped_grad, var), global_step=tf.train.get_or_create_global_step()
            )
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=tf.group([train, update_ops]),
                scaffold=scaffold,
            )
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {"tp": tf.metrics.mean(tp)}
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=eval_metric_ops
            )
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"predictions": predictions}
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
        d = d.repeat()
        if is_training:
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
    tf.set_random_seed(1234)
    tf.logging.set_verbosity(tf.logging.INFO)
    emb = np.load("embedding.npy")
    print(emb.size)
    contexts = np.load("all_context.npy")
    sample_size = 10
    model_fn = model_builder(emb, contexts, sample_size=sample_size)
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
            batch_size=30,
            sample_size=sample_size,
            total_context=len(contexts),
        ),
        steps=1,
        start_delay_secs=0,
        throttle_secs=0,
    )
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn_builder(
            input_file="train.tfrecord",
            is_training=True,
            batch_size=30,
            sample_size=sample_size,
            total_context=len(contexts),
        ),
        max_steps=200_000,
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    tf.app.run()
