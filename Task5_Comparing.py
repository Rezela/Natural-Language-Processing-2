# Your code for Task 5 here
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd

# Define documents
doc_A = "The cat sat on the mat."
doc_B = "The cat sat on the mat. The dog chased the cat."
doc_C = "The rocket launched into space."
corpus = [doc_A, doc_B, doc_C]

# Vectorize the documents
vectorizer = CountVectorizer()
text = vectorizer.fit_transform(corpus)

# Calculate similarities and distances
cos_sim = cosine_similarity(text)
euclid_distance = euclidean_distances(text)

# Display results in a DataFrame
cos_df = pd.DataFrame(cos_sim, index=["doc_A", "doc_B", "doc_C"], columns=["doc_A", "doc_B", "doc_C"])
euclid_df = pd.DataFrame(euclid_distance, index=["doc_A", "doc_B", "doc_C"], columns=["doc_A", "doc_B", "doc_C"])

print("Cosine Similarity Matrix:")
print(cos_df)
print("\nEuclidean Distance Matrix:")
print(euclid_df)