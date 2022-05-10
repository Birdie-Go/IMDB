import numpy as np
import io
from tqdm import tqdm
from tqdm.std import trange

def load(path):
    print("\n\nLoading...", end = '\n')
    f = io.open(path, 'r', encoding = 'utf-8', newline = '\n')
    n_word, dim = map(int, f.readline().split())
    print(f"{n_word} in Vector and the dim is {dim}.", end = '\n')
    word2vec = {}
    for line in tqdm(f):
        part = line.rstrip().split(' ')
        word2vec[part[0]] = np.array(list(map(float, part[1:])))
    return int(dim), word2vec

def calc_word_notin(all_word, word2vec):
    all_word = set(all_word)
    word_notin = []
    for word in all_word:
        if word not in word2vec:
            word_notin.append(word)
    # print(word_notin)
    return word_notin

def get_word2vec(list_words, dim, word2vec, word_notin, all_word):
    word2vec_list = np.zeros((len(list_words), dim))
    word_notin = set(word_notin)
    print("\n\nGet word2vec feature...", end = '\n')
    for j in trange(len(list_words)):
        word_list = [i for i in list_words[j] if i in all_word and i not in word_notin]
        for i in range(dim):
            value = 0.0
            for word in word_list:
                value += word2vec[word][i]
            value /= len(word_list) + 1
            word2vec_list[j][i] = value
    return word2vec_list
