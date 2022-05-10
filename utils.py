import re
import csv
from matplotlib.pyplot import disconnect
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from tqdm import tqdm
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

def clear_character(dict):
    '''只保留中英文、数字和空格，去掉其他东西'''
    pattern = re.compile("[^a-z^ ]")
    new_dict = []
    for i in dict:
        i = re.sub(pattern, '', i)
        new_dict.append(i)
    return new_dict

def delete_stopword(dict):
    '''去除停用词'''
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 's', 't', 'can', 'will', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y']
    filtered_dict = [w for w in dict if not w in stop_words]
    return filtered_dict


def clear(dict, features):
    '''文本清洗'''
    dict = dict.replace("<br />", " ").lower()
    dict = dict.split(' ')
    dict = delete_stopword(dict)
    dict = clear_character(dict)
    if features == 'word2vec':
        return dict
    else:
        porter_stemmer = PorterStemmer()
        new_dict = [porter_stemmer.stem(i) for i in dict]
        return new_dict

def load_dataset(name, features):
    '''加载数据集'''
    f = open('IMDB Dataset.csv', 'r', encoding='utf-8')
    dataset = []
    cnt = 0
    with f:
        reader = csv.reader(f, delimiter=",")
        print("Loading dateset...", end = '\n')
        for row in tqdm(reader):
            if row[0] == 'review':
                continue
            cnt = cnt + 1
            if name == 'debug':
                if cnt > 100:
                    break
            text = clear(row[0], features)
            label = row[1]
            dataset.append([text, label])
    return dataset

def split_dataset(dataset):
    text = []
    label = []
    for i in dataset:
        text.append(i[0])
        label.append(i[1])
    return text, label

def getid(all_word):
    id = defaultdict(int)
    cnt = 0
    for i in all_word:
        id[i] = cnt
        cnt += 1
    return id

def filter_list(list_words):
    # 总词频统计
    doc_frequency = defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i] += 1
    
    filter_word = []
    all_word = []
    low_frequency = 10
    high_frequency = 1000
    print("Begin to calc the word frequency : ", end = '\n')
    for i in tqdm(doc_frequency):
        if doc_frequency[i] < low_frequency or doc_frequency[i] > high_frequency:
            filter_word.append(i)
        elif i not in all_word:
            all_word.append(i)
    
    return set(filter_word), set(all_word)