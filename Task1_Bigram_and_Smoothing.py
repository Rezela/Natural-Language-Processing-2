# Your code for Task 1 here
import numpy as np
from collections import Counter, defaultdict

# Provided Corpus
train_corpus = [["<S>", "i", "am", "sam", "</S>"], ["<S>", "sam", "i", "am", "</S>"], ["<S>", "i", "do", "not", "like", "green", "eggs", "and", "ham", "</S>"]]
test_sentence = ["<S>", "i", "like", "green", "ham", "</S>"]

def train_bigram_model(corpus):
    unigram_counts = Counter()  # 计数器
    bigram_counts = Counter()
    vocab = set()  # Vocabulary set集合 自动去重

    for sentence in corpus:
        for i in range(len(sentence)):
            unigram_counts[sentence[i]]+=1
            vocab.add(sentence[i])
            # if not first word -> bigram
            if i > 0:
                bigram = (sentence[i-1], sentence[i])
                bigram_counts[bigram] += 1
    vocab_size = len(vocab)
    return unigram_counts, bigram_counts, vocab_size

# Calculate smoothed probability for one bigram(given 1 prev_word and 1 current word)
def calculate_bigram_prob(prev_word, word, unigram_counts, bigram_counts, V):
    # Implement smoothed probability calculation
    bigram = (prev_word, word)
    bigram_count = bigram_counts[bigram]
    unigram_count = unigram_counts[prev_word]
    # Laplace (Add-One) smoothing
    prob = (bigram_count + 1) / (unigram_count + V)
    return prob

def calculate_perplexity(sentence, unigram_counts, bigram_counts, V):
    # Implement perplexity calculation
    '''
    first: logPP(W) = (-1/N)*sum(log(p(w i|w i-1)))
    then: PP(W) = exp(logPP(w))
    '''
    N = len(sentence) - 1
    sum_prob = 0
    for i in range(N):
        bigram = (sentence[i],sentence[i+1])
        bigram_prob = calculate_bigram_prob(bigram[0], bigram[1], unigram_counts, bigram_counts, V)
        sum_prob += np.log(bigram_prob)
    perplexity = np.exp(-sum_prob/N)
    return perplexity

if __name__ == '__main__':
    # Train the model
    unigram_counts, bigram_counts, V = train_bigram_model(train_corpus)

    # Calculate and print perplexity
    perplexity = calculate_perplexity(test_sentence, unigram_counts, bigram_counts, V)
    print(f"Perplexity of the test sentence: {perplexity:.2f}")
