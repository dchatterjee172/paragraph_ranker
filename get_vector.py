import tensorflow as tf
from rank import _extractor
import numpy as np
import json

q = tf.placeholder(shape=(None, 20), name="q", dtype=tf.int32)
context = tf.placeholder(shape=(None, 300), name="context", dtype=tf.int32)

embedding = np.load("embedding.npy")
with open("word_to_index.json") as f:
    word_vecs = json.load(f)
embedding = tf.get_variable(
    "embedding", shape=embedding.shape, trainable=False, dtype=tf.float32
)

q_emb = tf.nn.embedding_lookup(embedding, q)
context_emb = tf.nn.embedding_lookup(embedding, context)
with tf.variable_scope("q"):
    q_vector = _extractor(
        _input=q_emb,
        num_vector=2,
        h_size=150,
        is_training=False,
        pos=None,
        k_size=4,
        strides=1,
    )

with tf.variable_scope("c"):
    c_vector = _extractor(
        _input=context_emb, num_vector=2, h_size=150, is_training=False, pos=None
    )
sess = tf.Session()
tf.train.init_from_checkpoint("model.ckpt-200000", {"/": "/"})
sess.run(tf.initializers.global_variables())


def get_para_vec(token_list):
    for tokens in token_list:
        for i in range(len(tokens)):
            tokens[i] = word_vecs.get(tokens[i], 1)
            if len(tokens) < 300:
                tokens.extend([0] * (300 - len(tokens)))
    feed_dict = {context: np.array(token_list)}
    return sess.run(c_vector, feed_dict)


def get_q_vec(token_list):
    for tokens in token_list:
        for i in range(len(tokens)):
            tokens[i] = word_vecs.get(tokens[i], 1)
            if len(tokens) < 20:
                tokens.extend([0] * (20 - len(tokens)))
    feed_dict = {q: np.array(token_list)}
    return sess.run(q_vector, feed_dict)


get_para_vec([["hi"] * 300] * 50)
get_q_vec([["hi"] * 20])
