from squad_df import v1
import spacy
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import json
from collections import OrderedDict as od


def get_word2vec(word_set):
    glove_path = "/home/dj/data/glove/glove.840B.300d.txt"
    word2vec_dict = {}
    with open(glove_path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=int(2.2e6), desc="getting wordvec"):
            array = line.lstrip().rstrip().split(" ")
            word = " ".join(array[0 : len(array) - 300])
            vector = list(map(float, array[-300:]))
            if word in word_set:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_set:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_set:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_set:
                word2vec_dict[word.upper()] = vector

    print(
        "{}/{} of word vocab have corresponding vectors in {}".format(
            len(word2vec_dict), len(word_set), glove_path
        )
    )
    return word2vec_dict


class FeatureWriter(object):
    def __init__(self, filename, wiki_start_end):
        self.filename = filename
        self.num_features = 0
        self.wiki_start_end = wiki_start_end
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values))
            )
            return feature

        features = dict()
        features["q"] = create_int_feature(feature[0])
        features["context_id"] = create_int_feature([feature[1]])
        features["unique_id"] = create_int_feature([feature[2]])
        features["q_len"] = create_int_feature([feature[3]])
        features["wiki_start_end"] = create_int_feature(self.wiki_start_end[feature[4]])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


v1 = sorted(v1, key=lambda x: x["wiki_idx"])
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])
all_context_id = dict()
id_to_q = dict()
all_context = list()
all_words = set()
data = [[], []]
context_len = []
question_len = []
wiki_start_end = od()
for i, sample in tqdm(enumerate(v1), total=98169, desc="processing"):
    context_text = sample["context"]
    if sample["wiki_idx"] not in wiki_start_end:
        try:
            last_wiki_id = next(reversed(wiki_start_end))
            wiki_start_end[last_wiki_id].append(len(all_context) - 1)
        except StopIteration:
            pass
        wiki_start_end[sample["wiki_idx"]] = [len(all_context)]
    if context_text not in all_context_id:
        context = [token.text for token in nlp(context_text)]
        if len(context) > 300:
            continue
        all_words.update(context)
        all_context_id[context_text] = len(all_context)
        all_context.append(context)
        context_len.append(len(context))
    question = [token.text for token in nlp(sample["question"])]
    if len(question) > 20:
        continue
    all_words.update(question)
    id_to_q[i] = sample["question"]
    data[int(sample["is_train"])].append(
        [question, all_context_id[context_text], i, len(question), sample["wiki_idx"]]
    )
last_wiki_id = next(reversed(wiki_start_end))
wiki_start_end[last_wiki_id].append(len(all_context) - 1)

print(len(data[0]), len(data[1]))

np.random.shuffle(data[0])
np.random.shuffle(data[1])

word_vecs = get_word2vec(all_words)
del all_words
embedding_mat = list()
word_to_index = dict()

for c in tqdm(all_context):
    for t in range(len(c)):
        token = c[t]
        if token not in word_to_index and token in word_vecs:
            c[t] = len(word_to_index) + 2
            word_to_index[token] = len(word_to_index) + 2
            embedding_mat.append(word_vecs[token])
        elif token not in word_vecs:
            c[t] = 1
        else:
            c[t] = word_to_index[token]
    if len(c) < 300:
        c += [0] * (300 - len(c))

writers = (
    FeatureWriter("test.tfrecord", wiki_start_end),
    FeatureWriter("train.tfrecord", wiki_start_end),
)

for i, d in enumerate(data):
    for q in tqdm(d):
        for t in range(len(q[0])):
            token = q[0][t]
            if token not in word_to_index and token in word_vecs:
                q[0][t] = len(word_to_index) + 2
                word_to_index[token] = len(word_to_index) + 2
                embedding_mat.append(word_vecs[token])
            elif token not in word_vecs:
                q[0][t] = 1
            else:
                q[0][t] = word_to_index[token]
        if len(q[0]) < 20:
            q[0] += [0] * (20 - len(q[0]))
        writers[i].process_feature(q)

map(lambda x: x.close(), writers)

embedding_mat.insert(0, [0] * 300)
embedding_mat.insert(1, list(np.random.uniform(-0.05, 0.05, (300,))))
np.save("embedding", np.array(embedding_mat))
np.save("all_context", np.array(all_context))
np.save("context_len", np.array(context_len))
with open("id_to_context.json", "w") as f:
    json.dump({v: k for k, v in all_context_id.items()}, f, indent=4),
with open("id_to_question.json", "w") as f:
    json.dump(id_to_q, f, indent=4)
with open("word_to_index.json", "w") as f:
    json.dump(word_to_index, f, indent=4)
