# Task 4: Implement Viterbi for a simple POS HMM (skeleton)
import math
from typing import List, Dict, Tuple

# Define your tag set
TAGS = ["DET", "NOUN", "VERB", "PRT"]

# Define HMM parameters (fill using the given parameters)
pi: Dict[str, float] = {
    "DET": 0.50,
    "NOUN": 0.20,
    "VERB": 0.20,
    "PRT": 0.10
}

A: Dict[str, Dict[str, float]] = {
    "DET":  {"DET": 0.05, "NOUN": 0.75, "VERB": 0.15, "PRT": 0.05},
    "NOUN": {"DET": 0.05, "NOUN": 0.10, "VERB": 0.75, "PRT": 0.10},
    "VERB": {"DET": 0.10, "NOUN": 0.35, "VERB": 0.40, "PRT": 0.15},
    "PRT":  {"DET": 0.05, "NOUN": 0.10, "VERB": 0.75, "PRT": 0.10}
}

B: Dict[str, Dict[str, float]] = {
    "DET":  {"the": 0.80, "a": 0.20},
    "NOUN": {"book": 0.45, "table": 0.25, "flight": 0.20, "i": 0.05, "on": 0.05},
    "VERB": {"is": 0.40, "want": 0.35, "book": 0.20, "to": 0.03, "on": 0.02},
    "PRT":  {"to": 0.70, "on": 0.30}
}

UNK = 1e-8

# Return log-probability for emitting 'word' from 'tag'. Use UNK for unseen words.
def emission_logprob(tag: str, word: str) -> float:
    prob = B.get(tag, {}).get(word, UNK)
    return math.log(prob)

# Implement Viterbi in log-space
# 1) initialize  2) dynamic programming with backpointers  3) termination + backtrace
def viterbi(tokens: List[str]) -> Tuple[List[str], float]:
    n = len(tokens)
    V = [{} for _ in range(n)]   # DP table
    backpointer = [{} for _ in range(n)]

    # Initialization
    for tag in TAGS:
        if pi[tag] > 0:
            V[0][tag] = math.log(pi[tag]) + emission_logprob(tag, tokens[0])
        else:
            V[0][tag] = -math.inf
        backpointer[0][tag] = None

    # Recurrence
    for i in range(1, n):
        for tag in TAGS:
            max_prob, best_prev = -math.inf, None
            for prev in TAGS:
                if V[i-1][prev] == -math.inf:
                    continue
                prob = V[i-1][prev] + math.log(A[prev][tag]) + emission_logprob(tag, tokens[i])
                if prob > max_prob:
                    max_prob, best_prev = prob, prev
            V[i][tag] = max_prob
            backpointer[i][tag] = best_prev

    # Termination
    final_tag = max(V[n-1], key=V[n-1].get)
    best_logprob = V[n-1][final_tag]

    # Backtrace
    tags = [final_tag]
    for i in range(n-1, 0, -1):
        tags.insert(0, backpointer[i][tags[0]])

    return tags, best_logprob

# Prepare the two sentences (lowercased tokens)
sentence1 = ["the", "book", "is", "on", "the", "table"]
sentence2 = ["i", "want", "to", "book", "a", "flight"]

# Run your decoder and print outputs
tags1, logp1 = viterbi(sentence1)
print(list(zip(sentence1, tags1)), " | logP=", round(logp1, 3))
tags2, logp2 = viterbi(sentence2)
print(list(zip(sentence2, tags2)), " | logP=", round(logp2, 3))
