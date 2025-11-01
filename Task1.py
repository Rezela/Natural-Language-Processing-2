# Task 1: Implementing a Bigram Language Model with Laplace Smoothing

import numpy as np
from collections import Counter

# 提供的训练语料和测试句子
train_corpus = [
    ["<S>", "i", "am", "sam", "</S>"],
    ["<S>", "sam", "i", "am", "</S>"],
    ["<S>", "i", "do", "not", "like", "green", "eggs", "and", "ham", "</S>"]
]
test_sentence = ["<S>", "i", "like", "green", "ham", "</S>"]

# 1. 训练函数：统计unigram和bigram
def train_bigram_model(corpus):
    unigram_counts = Counter()
    bigram_counts = Counter()
    vocab = set()

    for sentence in corpus:
        for i in range(len(sentence)):
            unigram_counts[sentence[i]] += 1
            vocab.add(sentence[i])
            if i > 0:
                bigram = (sentence[i - 1], sentence[i])
                bigram_counts[bigram] += 1

    V = len(vocab)  # 词汇表大小
    return unigram_counts, bigram_counts, V

# 2. 计算带Laplace平滑的bigram概率
def calculate_bigram_prob(prev_word, word, unigram_counts, bigram_counts, V):
    bigram = (prev_word, word)
    bigram_count = bigram_counts[bigram]
    unigram_count = unigram_counts[prev_word]
    prob = (bigram_count + 1) / (unigram_count + V)
    return prob

# 3. 计算困惑度
def calculate_perplexity(sentence, unigram_counts, bigram_counts, V):
    N = len(sentence) - 1  # bigram数量
    log_prob_sum = 0.0

    for i in range(1, len(sentence)):
        prev_word = sentence[i - 1]
        word = sentence[i]
        prob = calculate_bigram_prob(prev_word, word, unigram_counts, bigram_counts, V)
        log_prob_sum += np.log(prob)

    perplexity = np.exp(-log_prob_sum / N)
    return perplexity

# 4. 训练并评估
unigram_counts, bigram_counts, V = train_bigram_model(train_corpus)
perplexity = calculate_perplexity(test_sentence, unigram_counts, bigram_counts, V)

print(f"Perplexity of the test sentence: {perplexity:.2f}")
