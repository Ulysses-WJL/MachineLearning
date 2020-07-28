#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-11-16 10:43:41
@Description: 使用朴素贝叶斯对垃圾邮件进行分类
@LastEditTime: 2019-11-17 20:12:51
'''
from operator import itemgetter
import re
import os
import numpy as np
import pandas as pd
import feedparser
"""
1. 收集数据  email下 spam和ham
2. 准备数据  将文本文件切割为词条向量
3. 分析数据  检查词条确保解析的正确性
4. 训练算法  使用之前的train_naive_bayes算法
5. 测试算法  classify函数  并构建新函数测试错误率
6. 使用算法
"""

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
EMAIL_PATH_HAM = os.path.join(CURRENT_DIR, 'email', 'ham')
EMAIL_PATH_SPAM = os.path.join(CURRENT_DIR, 'email', 'spam')
STOP_WORDS_PATH = os.path.join(CURRENT_DIR, 'stopword.txt')

# 收集数据
def load_data_set():
    """
    创建数据集,把文本看成单词向量或词条向量
    Returns:
        DataFrame: 单词列表 是否为侮辱性
    """
    # 文本 拆分为一条条句子 再拆分一个个单词
    posting = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive 0 not
    return posting, class_vec

def create_vocab_list(data_set):
    """
    获取所有单词的集合
    :param data_set: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocab_set = set()  # create empty set
    for item in data_set:
        # | 求两个集合的并集
        vocab_set = vocab_set | set(item)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocab_list: 所有单词集合列表
    :param input_set: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    result = [0] * len(vocab_list)
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
    return result


def normalize(data_set, all_tokens):
    """
    将data_set中的数据整理为 便于处理的int型数据
    """
    for i in range(len(data_set)):
        data_set[i] = set_of_words2vec(all_tokens, data_set[i])


def text(file):
    """
    准备数据 切分文本
    Args:
        file ([type]): 邮件文件地址
    Returns:
        [type]: 邮件中单词表(长度>2, 都转成小写)
    """
    with open(file, encoding='windows-1252') as email_f:
        # 使用非单词()进行划分
        tokens = re.split(r"\W+", email_f.read())
        return  [token.lower() for token in tokens if len(token) > 2]


def train_naive_bayes(train_mat, train_category):
    """
    训练算法
    Args:
        train_df (pd.DataFrame):文件单词dataframe
    Returns:
        [type]: 类条件概率(似然), 先验概率
    """
    # 样本数量
    train_doc_num = len(train_mat)
    # 所有的单词数量
    words_num = len(train_mat[0])
    p_priori = (np.sum(train_category) + 1) / (train_doc_num + 2)

    p0_num = np.ones(words_num)
    p1_num = np.ones(words_num)

    # 类条件概率 拉普拉斯修正后 P(xi|c) = (Dc,xi + 1) / (Dc + Ni)
    p0num_all = 2.0
    p1num_all = 2.0

    for i in range(train_doc_num):
        if train_category[i] == 1:
            p1_num += train_mat[i]  # p1类别属性出现次数表 [1, 2, 1, ...]
            p1num_all += np.sum(train_mat[i])  # p1类别属性总数
        else:
            p0_num += train_mat[i]
            p0num_all += np.sum(train_mat[i])
    p1_vec = np.log(p1_num / p1num_all)
    p0_vec = np.log(p0_num / p0num_all)
    return p1_vec, p0_vec, p_priori


def classify(input_data_vec, p1_vec, p0_vec, p_priori):

    # P(c|x) = P(C) * P(x|c) / P(X)
    # 取log后 乘变加
    p1 = np.sum(input_data_vec * p1_vec) + np.log(p_priori)

    p0 = np.sum(input_data_vec * p0_vec) + np.log(1 - p_priori)

    if p1 > p0:
        return 1
    else:
        return 0

# -------------------使用朴素贝叶斯对是否侮辱性文本进行区分--------------------------

def test_naive_bayes():
    """
    测试朴素贝叶斯算法
    """
    result = {1:"侮辱性的", 0:"非侮辱性的"}
    # 加载数据
    list_post, list_classes = load_data_set()
    # 2. 创建单词集合
    vocab_list = create_vocab_list(list_post)

    # 3 准备数据 按是否出现过, 进行转换
    normalize(list_post, vocab_list)

    # 4. 训练数据
    p1v, p0v, priori = train_naive_bayes(
        np.array(list_post), np.array(list_classes))

    data_1 = ['my', 'dog', 'stupid', 'fuck']


    ret = classify(set_of_words2vec(vocab_list, data_1), p1v, p0v, priori)
    print(f'{data_1}的测试结果为: {result[ret]}!')

    data_2 = ['love', 'my', 'dalmation']
    ret = classify(set_of_words2vec(vocab_list, data_2), p1v, p0v, priori)
    print(f'{data_2}的测试结果为: {result[ret]}!')

# ----------------------分辨是否为垃圾邮件-----------------------

def spam_test():

    ham_spam_list, full_tokens, test_list, test_cat = [], [], [], []
    p_priori = [0, 0]
    category_list = []
    # 正常邮件
    for ham_file in os.listdir(EMAIL_PATH_HAM):
        words = text_parse(os.path.join(EMAIL_PATH_HAM, ham_file))
        # [文件1单词list, 文件2单词list]
        ham_spam_list.append(words)
        full_tokens.extend(words)  # 所有单词list
        category_list.append(1)

    # 垃圾邮件
    for spam_file in os.listdir(EMAIL_PATH_HAM):
        words = text(os.path.join(EMAIL_PATH_SPAM, spam_file))
        # [文件1单词list, 文件2单词list]
        ham_spam_list.append(words)
        full_tokens.extend(words)  # 所有单词list
        category_list.append(0)


    full_tokens = list(set(full_tokens))

    normalize(ham_spam_list, full_tokens)

    # print(ham_spam_list)

    # 交叉验证 取10个为测试集 其余为训练集l
    for _ in range(10):
        index = int(np.random.uniform(0, len(ham_spam_list)))
        test_list.append(ham_spam_list.pop(index))
        test_cat.append(category_list.pop(index))
    # ham, spam分类的类条件概率  先验概率
    likelihood_1, likelihood_0,  p_priori = train_naive_bayes(
        np.array(ham_spam_list), np.array(category_list))
    error_count = 0
    # last = np.array(test_list[-1])
    # last_email = np.array(full_tokens)[np.where(last > 0)]
    # print(last_email)
    for i in range(10):
        label = classify(test_list[i], likelihood_1, likelihood_0, p_priori)
        if label != test_cat[i]:
            error_count += 1
    print(f'错误率: {error_count/10:.2%}')

# ---------------------区分不同类别的news------------------------
"""
我们将每个词的出现与否作为一个特征，这可以被描述为 词集模型(set-of-words model)。
如果一个词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息，
这种方法被称为 词袋模型(bag-of-words model)。在词袋中，每个单词可以出现多次，
而在词集中，每个词只能出现一次
"""
def bag_words2vec(vocab_list, input_set):
    # 注意和原来的做对比  词袋模型 bag-of-words 一个单词可以出现多次
    result = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] += 1
        else:
            pass
            # print('the word: {} is not in my vocabulary'.format(word))
    return result

def get_stop_words(file_path):
    with open(file_path) as f:
        words = list(map(lambda x:x.strip(), f.readlines()))
    # print(len(words), words[-5:])
    return words

def cal_most_freq(vocab_list, full_text):
    """
    计算出现频率
    Args:
        vocab_list ([type]): 全部的单词集合
        full_text ([type]): 一条信息中的单词
    Returns:
        [type]: 出现频率最高的30个单词
    """
    freq_dict = {}

    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    # #根据每个词出现的次数低到高对字典进行排序
    sorted_freq = sorted(freq_dict.items(), key=itemgetter(1), reverse=True)
    return sorted_freq[:30]  # 取最高的30个单词

"""d = feedparser.parse()
每个RSS和Atom订阅源都包含一个标题（d.feed.title）和一组文章条目(d.entries)
通常每个文章条目都有一段摘要（d.entries[i].summary）,
或者是包含了条目中实际文本的描述性标签（d.entries[i].description）

d.entries 该属性类型为列表，表示一组文章的条目


d.feed  feed 对应的值也是一个字典
"""

def text_parse(text):
    tokens = re.split(r"\W+", text)
    return  [token.lower() for token in tokens if len(token) > 2]

def classify_news(feed0, feed1):
    """
    根据RSS源获取的新闻,辨别种类
    Args:
        feed0 ([type]): 种类1
        feed1 ([type]): 种类2
    """
    doc_list, class_list, full_text = [], [], []
    # 取相同数量的RSS源信息
    min_len = min(len(feed0['entries']), len(feed1['entries']))
    print(min_len)
    for i in range(min_len):
        # 获取摘要信息
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

    # 包含所有单词的list, 无重复
    vocab_list = create_vocab_list(doc_list)
    print(f"词汇表长度{len(vocab_list)}")

    # 先去除stopwords
    for word in get_stop_words(STOP_WORDS_PATH):
        if word in vocab_list:
            vocab_list.remove(word)
    print(f"去除stopwords后的词汇表长度{len(vocab_list)}")
    # 去除出现率最高的30个单词
    top30_words = cal_most_freq(vocab_list, full_text)
    for word in top30_words:
        vocab_list.remove(word[0])

    test_mat = []
    test_class = []
    # 单词出现次数矩阵
    for i in range(len(doc_list)):
        doc_list[i] = bag_words2vec(vocab_list, doc_list[i])

    # 取20 个为测试集
    for _ in range(20):
        index = int(np.random.uniform(0, len(doc_list)))
        test_mat.append(doc_list.pop(index))
        test_class.append(class_list.pop(index))

    p1v, p0v, priori = train_naive_bayes(
        np.array(doc_list), np.array(class_list))

    error_conut = 0
    for i in range(20):
        label = classify(np.array(test_mat[i]), p1v, p0v, priori)
        if label != test_class[i]:
            error_conut += 1
    print(f"错误率:{error_conut / 20:.2%}")
    return vocab_list, p1v, p0v


def test_feedparse():
    # 最具表征性的词汇显示函数
    url1 = 'http://sports.yahoo.com/rss/'
    url2 = "http://feeds.sciencedaily.com/sciencedaily"
    url3 = 'https://finance.yahoo.com/rss/'

    d_sports = feedparser.parse(url1)
    d_finace = feedparser.parse(url3)

    vocab_list, p1v, p0v = classify_news(d_sports, d_finace)

    top1, top0 = [], []
    for i in range(len(p1v)):
        if p1v[i] > -6.0:
            top1.append((vocab_list[i], p1v[i]))
        if p0v[i] > -6.0:
            top0.append((vocab_list[i], p0v[i]))
    print(url3)
    top1.sort(key=lambda x: x[1], reverse=True)
    print(top1)
    print(url1)
    top0.sort(key=lambda x: x[1], reverse=True)
    print(top0)

if __name__ == "__main__":
    # test_naive_bayes()
    # spam_test()

    test_feedparse()


