# -*- coding: utf-8 -*-
from collections import defaultdict
import math
from tqdm import tqdm
from tqdm.std import trange
from utils import *
import numpy as np

def get_tfidf(all_word, id, list_words, feature):
    word_tf = []
    print("\n\nBegin to calc the tf: ", end = '\n')
    for word_list in tqdm(list_words):
        one_frequency = defaultdict(int)
        for i in word_list:
            if i in all_word:
                one_frequency[i] += 1
        one_tf = {}
        for i in one_frequency:
            one_tf[i] = one_frequency[i] / sum(one_frequency.values())
        word_tf.append(one_tf)
    
    if feature == 'tf-idf':
        # 计算每个词的 IDF 值
        doc_num = len(list_words)
        word_idf = {} # 存储每个词的 idf 值
        word_doc = defaultdict(int) # 存储包含该词的文档数
        print("\n\nBegin to calc the idf : ", end = '\n')
        for i in tqdm(list_words):
            i = set(i)
            for word in i:
                if word in all_word:
                    word_doc[word] += 1
        for i in all_word:
            word_idf[i] = math.log(doc_num / (word_doc[i] + 1))

        print("\n\nBegin to calc the tf-idf : ", end = '\n')
        tfidf_list = np.zeros((doc_num, len(all_word)))
        for i in tqdm(range(doc_num)):
            word_list= set(list_words[i])
            for word in set(word_tf[i]):
                tfidf_list[i][id[word]] = word_tf[i][word] * word_idf[word]
        return tfidf_list
    elif feature == 'tf':
        doc_num = len(list_words)
        tf_list = np.zeros((doc_num, len(all_word)))
        for i in trange(doc_num):
            for word in set(word_tf[i]):
                tf_list[i][id[word]] = word_tf[i][word]
        return tf_list