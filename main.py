from typing import Text
from utils import *
import tfidf
import word2vec
import fbnn
from tqdm import tqdm
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--feature", type=str, default="tf-idf")
args = parser.parse_args()
features = args.feature
mode = "fbnn"
epochs = 100
active_function = 'sigmoid'
hidden_n = 10

# load dataset
n_train = 30001
n_test = 10000
dataset = load_dataset("test", features)
text, label = split_dataset(dataset)
train_text, train_label = split_dataset(dataset[ : n_train])
test_text, test_label = split_dataset(dataset[n_train : n_train + n_test])
value_text, value_label = split_dataset(dataset[n_train + n_test : ])

# get word dictionary
# if not os.path.exists(".\\Checkpoint\\tf-idf_feature.npy"):
#     filter_word, all_word = filter_list(text)
#     print("filter word : %d ; useful word : %d " % (len(filter_word), len(all_word)), end = '\n')
#     id = getid(all_word)
#     np.save("./Checkpoint/all_word", all_word)
#     np.save("./Checkpoint/word_id", id)
# else:
#     id = np.load(".\\Checkpoint\\word_id.npy")
#     all_word = np.load(".\\Checkpoint\\all_word.npy")
filter_word, all_word = filter_list(text)
print("filter word : %d ; useful word : %d " % (len(filter_word), len(all_word)), end = '\n')
id = getid(all_word)

# get features
if features == "tf-idf" or features == "tf":
    word_features = tfidf.get_tfidf(all_word, id, train_text, features)
    # print("Saving...", end = '\n')
    # np.save("./Checkpoint/" + features + "_feature", word_features)
elif features == 'word2vec':
    dim, data = word2vec.load("wiki-news-300d-1M.vec")
    word_notin = word2vec.calc_word_notin(all_word, data)
    cnt_notin = len(word_notin)
    print(f"\n\n{cnt_notin} words not in.", end = '\n')
    word_features = word2vec.get_word2vec(train_text, dim, data, word_notin, all_word)
    # print("Saving...", end = '\n')
    # np.save("./Checkpoint/word2vec_feature", word_features)

# train
if features == "tf-idf" or features == "tf":
    model = fbnn.BPNeuralNetwork(len(all_word), hidden_n, features)
elif features == 'word2vec':
    model = fbnn.BPNeuralNetwork(dim, hidden_n, features)
train_label_num = np.array([0 if train_label[i] == 'positive' else 1 for i in range(len(train_label))], dtype = int)

# value
value_label_num = np.array([0 if value_label[i] == 'positive' else 1 for i in range(len(value_label))], dtype = int)
if features == "tf-idf" or features == "tf":
    value_tfidf = tfidf.get_tfidf(all_word, id, value_text, features)
    model.train(word_features, train_label_num, epochs, value_tfidf, value_label_num)
elif features == 'word2vec':
    value_word2vec = word2vec.get_word2vec(value_text, dim, data, word_notin, all_word)
    model.train(word_features, train_label_num, epochs, value_word2vec, value_label_num)
