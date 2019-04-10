import tensorflow as tf
import numpy as np
import json
from collections import defaultdict
from opt import AdamWeightDecayOptimizer

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "./tmp", "Model Directory")
flags.DEFINE_bool("train", True, "train?")
flags.DEFINE_bool("eval", True, "eval?")
flags.DEFINE_integer("sample_size", 10, "sample size")
flags.DEFINE_integer("top_k", 10, "checking if correct para is in top k")


def model_builder(embedding_, context_, sample_size):
    num_units = 128
    num_vector = 8
    pos_embedding_ = np.ones((300, 300), dtype=np.float32)
    ls = 151
    le = num_units // num_vector
    for k in range(1, le):
        for j in range(1, ls):
            pos_embedding_[j - 1, k - 1] = (1.0 - j / ls) - (k / le) * (
                1.0 - 2.0 * j / ls
            )
    pos_embedding_ = np.expand_dims(pos_embedding_, 0) / 2.0

    def _extractor(
        _input,
        num_vector,
        h_size,
        is_training,
        batch_size,
        pos,
        k_size=7,
        strides=2,
        sample_size=sample_size,
    ):
        all_input_ = []
        for i in range(num_vector):
            input_ = tf.layers.separable_conv1d(
                _input,
                h_size,
                k_size,
                padding="same",
                strides=strides,
                activation=tf.nn.leaky_relu,
            )
            seq_len = tf.shape(input_)[-2]
            input_ = input_ + pos[:, :seq_len, :]
            input_ = tf.layers.batch_normalization(input_, training=is_training)
            score = tf.nn.softmax(
                tf.matmul(input_, input_, transpose_b=True)
                / tf.sqrt(tf.constant(h_size, dtype=tf.float32))
            )
            p = tf.matmul(score, input_)
            p = tf.layers.dense(
                tf.reshape(p, [-1, h_size]),
                1,
                kernel_initializer=tf.glorot_normal_initializer,
            )
            p = tf.nn.softmax(tf.reshape(p, [-1, seq_len]))
            p = tf.expand_dims(p, -2)
            input_ = tf.squeeze(tf.matmul(p, input_), -2)
            all_input_.append(input_)

        input_ = tf.concat(all_input_, -1)
        input_ = tf.contrib.layers.layer_norm(
            input_, begin_norm_axis=-1, begin_params_axis=-1
        )
        input_ = tf.reshape(input_, [-1, num_vector, h_size])
        input_ = tf.layers.dropout(
            input_,
            0.5,
            noise_shape=[batch_size * sample_size, num_vector, 1],
            training=is_training,
        )
        input_ = tf.reshape(input_, [-1, num_vector * h_size])
        return input_

    def model(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        embedding = tf.get_variable(
            "embedding", shape=embedding_.shape, trainable=False, dtype=tf.float32
        )
        pos_embedding = tf.get_variable(
            "pos_embedding",
            shape=pos_embedding_.shape,
            trainable=False,
            dtype=tf.float32,
        )
        all_context = tf.get_variable(
            "all_context", shape=context_.shape, trainable=False, dtype=tf.int32
        )

        def init_fn(scaffold, sess):
            sess.run(embedding.initializer, {embedding.initial_value: embedding_})
            sess.run(
                pos_embedding.initializer, {pos_embedding.initial_value: pos_embedding_}
            )
            sess.run(all_context.initializer, {all_context.initial_value: context_})
            tf.logging.info("embedding initialized")

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(f"  name = {name}, shape = {features[name].shape}")
        context_id = features["context_id"]
        q = features["q"]
        batch_size = tf.shape(q)[0]
        q = tf.nn.embedding_lookup(embedding, q)
        context = tf.nn.embedding_lookup(all_context, context_id)
        context = tf.nn.embedding_lookup(embedding, context)
        with tf.variable_scope("q"):
            q = _extractor(
                q,
                num_vector,
                num_units // num_vector,
                is_training,
                batch_size,
                pos_embedding,
                k_size=4,
                strides=1,
                sample_size=1,
            )
        with tf.variable_scope("c"):
            context = tf.reshape(
                context, [batch_size * sample_size, -1, embedding_.shape[-1]]
            )
            context = _extractor(
                context,
                num_vector,
                num_units // num_vector,
                is_training,
                batch_size,
                pos_embedding,
            )
            context = tf.reshape(context, [batch_size, sample_size, -1])
        q = tf.expand_dims(q, -2)
        logits = tf.matmul(context, q, transpose_b=True) / tf.sqrt(
            tf.constant(num_units, dtype=tf.float32)
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
            """optimizer = AdamWeightDecayOptimizer(
                learning_rate=0.001,
                weight_decay_rate=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=[
                    "LayerNorm",
                    "layer_norm",
                    "bias",
                    "batch_norm",
                ],
            )"""
            var = tf.trainable_variables()
            grads = tf.gradients(loss, var)
            clipped_grad, norm = tf.clip_by_global_norm(grads, 1)
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
            predictions = {
                "predictions": predictions,
                "logits": logits,
                "context_id": context_id,
                "unique_id": features["unique_id"],
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    return model


def input_fn_builder(
    input_file, is_training, batch_size, sample_size, total_context, repeat=True
):
    arange = np.arange(0, total_context)

    def input_fn():
        name_to_features = {
            "context_id": tf.FixedLenFeature([1], tf.int64),
            "q": tf.FixedLenFeature([20], tf.int64),
            "unique_id": tf.FixedLenFeature([], tf.int64),
        }
        arange_tensor = tf.constant(arange)

        def _decode_record(record, name_to_features):
            example = tf.parse_single_example(record, name_to_features)
            context_id = example["context_id"][0]
            sample = tf.concat(
                [arange_tensor[:context_id], arange_tensor[context_id + 1 :]], axis=-1
            )
            sample = tf.random_shuffle(sample)[: sample_size - 1]
            example["context_id"] = tf.concat([example["context_id"], sample], axis=-1)
            return example

        d = tf.data.TFRecordDataset(input_file)
        drop_remainder = False
        if repeat:
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
    contexts = np.load("all_context.npy")
    with open("id_to_context.json") as f:
        id_to_context = json.load(f)
    with open("id_to_question.json") as f:
        id_to_q = json.load(f)
    print(emb.shape, contexts.shape)
    model_fn = model_builder(emb, contexts, sample_size=FLAGS.sample_size)
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=2,
        log_step_count_steps=100,
    )
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    if FLAGS.train:
        eval_spec = tf.estimator.EvalSpec(
            input_fn=input_fn_builder(
                input_file="test.tfrecord",
                is_training=False,
                batch_size=30,
                sample_size=FLAGS.sample_size,
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
                sample_size=FLAGS.sample_size,
                total_context=len(contexts),
            ),
            max_steps=200_000,
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    if FLAGS.eval:
        input_fn = input_fn_builder(
            input_file="test.tfrecord",
            is_training=False,
            batch_size=30,
            sample_size=FLAGS.sample_size,
            total_context=len(contexts),
            repeat=False,
        )
        tp = 0
        tp_top = 0
        count = 0
        res = defaultdict(dict)
        mrr = 0
        for result in estimator.predict(input_fn, yield_single_examples=True):
            count += 1
            pred = int(result["predictions"])
            logits = result["logits"]
            context_id = result["context_id"]
            unique_id = str(result["unique_id"])
            if pred == 0:
                tp += 1
                res[id_to_q[unique_id]]["tp"] = 1
            else:
                res[id_to_q[unique_id]]["tp"] = 0
                res[id_to_q[unique_id]][f"tp_top_{FLAGS.top_k}"] = 0
            res[id_to_q[unique_id]]["para"] = list()
            ranked = sorted(zip(context_id, logits), key=lambda x: x[1], reverse=True)
            for i, (c, l) in enumerate(ranked[: FLAGS.top_k]):
                if c == context_id[0]:
                    actual = True
                    tp_top += 1
                    res[id_to_q[unique_id]][f"tp_top_{FLAGS.top_k}"] = 1
                    mrr += 1 / (1 + i)
                else:
                    actual = False
                res[id_to_q[unique_id]]["para"].append(
                    {
                        "text": id_to_context[str(c)],
                        "is_actual": actual,
                        "logit": float(l),
                    }
                )
        print(f"tp {tp / count * 100}")
        print(f"tp_top_{FLAGS.top_k} {tp_top / count * 100}")
        print(f"mrr_{FLAGS.top_k} {mrr / count}")
        with open("failed.json", "w") as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    tf.app.run()
