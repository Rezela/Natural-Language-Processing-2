# Your code for Task 3 here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

document_corpus = [
    "The field of machine learning has seen rapid growth in recent years, especially in deep learning.",
    "Natural language processing allows machines to understand and respond to human text.",
    "Computer vision focuses on enabling computers to see and interpret the visual world.",
    "Deep learning models like convolutional neural networks are powerful for computer vision tasks.",
    "Recurrent neural networks are often used for sequential data in natural language processing.",
    "The advances in reinforcement learning have led to breakthroughs in game playing and robotics.",
    "Transfer learning enables models trained on large datasets to be adapted for new tasks with limited data.",
    "Unsupervised learning techniques can discover hidden patterns in data without labeled examples.",
    "Optimization algorithms such as stochastic gradient descent are crucial for training neural networks.",
    "Attention mechanisms have improved the performance of natural language translation and image captioning.",
    "Generative adversarial networks create realistic images and are used for data augmentation.",
    "Feature engineering and selection are important steps in classical machine learning pipelines.",
    "Object detection is a key task in computer vision that involves locating instances within images.",
    "The combination of convolutional and recurrent networks is used for video classification tasks.",
    "Zero-shot learning allows models to recognize objects and concepts they have not seen during training.",
    "Natural language generation is used for creating text summaries and chatbot responses.",
    "Graph neural networks leverage graph structures for tasks such as social network analysis and chemistry.",
    "Hyperparameter tuning can significantly improve the accuracy of deep learning models.",
    "Cross-modal learning involves integrating information from multiple data sources such as text and images.",
    "Evaluating model performance requires a good choice of metrics such as F1-score and RMSE."
]

# Create and fit the vectorizer
vectorizer = TfidfVectorizer()

# Transform the corpus to DTM
doc_term_matrix = vectorizer.fit_transform(document_corpus)

def rank_documents(query, vectorizer, doc_term_matrix, top_n=3):
    vec_query = vectorizer.transform([query])
    similarities = cosine_similarity(vec_query, doc_term_matrix).flatten()

    # Get indices of top_n documents, sorted by similarity
    top_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for rank, index in enumerate(top_indices, start=1):  # 从1开始为每个索引分配排名号
        results.append((rank, index, document_corpus[index], similarities[index]))
    return results

# Demonstrate with a sample query
query = "deep learning models for vision"
ranked_docs = rank_documents(query, vectorizer, doc_term_matrix, top_n=3)
print(f"Top {len(ranked_docs)} documents for the query: '{query}'\n")
for rank, index, document, similarity in ranked_docs:
    print(f"Rank {rank}: (the {index}th document) {document} | Score = {similarity:.4f}")
