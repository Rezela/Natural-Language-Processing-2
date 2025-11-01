# Task 2: Text Classification with TF-IDF and Naive Bayes
import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. 下载并加载数据集
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
dataset_zip = "smsspamcollection.zip"
dataset_file = "SMSSpamCollection"

# 如果本地没有数据集，就下载并解压
if not os.path.exists(dataset_file):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_zip)
    import zipfile
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset downloaded and extracted.")


# 读取数据
df = pd.read_csv(dataset_file, sep="\t", header=None, names=["label", "message"])
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# 2. Split the dataset (80% train, 20% test)
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Create a pipeline: TF-IDF + MultinomialNB
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

# 4. Train the pipeline
pipeline.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = pipeline.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Predict on new messages
new_messages = [
    "Congratulations! You've won a $1,000 gift card. Go to http://example.com to claim now.",
    "Hi mom, I'll be home for dinner tonight."
]

predictions = pipeline.predict(new_messages)

for msg, pred in zip(new_messages, predictions):
    print(f"Message: {msg}\nPredicted class: {pred}\n")
