# Your code for Task 2 here
import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset, e.g, as follows. But you may modify it.
# df = pd.read_csv('path_to_your_dataset/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# download dataset
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
dataset_zip = "smsspamcollection.zip"
dataset_file = "SMSSpamCollection"

if not os.path.exists(dataset_file):
    print("Downloading dataset...")
    # download
    urllib.request.urlretrieve(dataset_url, dataset_zip)
    # unzip
    import zipfile
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(".")  # unzip to current directory
    print("Dataset downloaded and extracted.")

# Load the dataset, Tab("\t") as separator, without a line of header, column 1: label, column 2: message
df = pd.read_csv(dataset_file, sep="\t", header=None, names=["label", "message"])
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# Split data
x = df["message"]
y = df["label"]

# set a random seed: 28, stratify=y to make sure test size has same proportion of each class as the whole dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=28, stratify=y
)

# Create and train the pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])

# Train the pipeline
pipeline.fit(x_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Predict on new messages
new_messages = [
    "Congratulations! You've won a $1,000 gift card. Go to http://example.com to claim now.",
    "Hi mom, I'll be home for dinner tonight."
]
predictions = pipeline.predict(new_messages)
for message, prediction in zip(new_messages, predictions):  # 两个列表并行配对遍历
    print(f"Message: {message}\nPrediction: {prediction}\n")