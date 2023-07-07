# Import SKlearn
import sklearn

# Import necessary modules from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Count Vectorizer: Help transform single words into numerical features. Numerical features are a sequence of numbers that corrospond to a word in the text corpus.
#                   This module creates a matrix that gives each word a number.
# train_test_split: This module helps split a dataset into training and testing datasets.
# MultinomialNB:    Naive Bayes in an algorythem for natural language processing. This module is sued in the case of discreate features for text classification, 
#                   including sentiment analysis.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data Preparation
#       Task: Create a list of single words and a list of corresponding sentiment labels that indicate whether the word is positive, negative, or neutral

# Create empty lists
words = []
labels = []

# Read dataset from csv file and assign data into corresponding list 
import csv

with open('dev_words.csv') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    lineCount = 0
    for row in csvReader:
        words.append(row[1])
        labels.append(row[2])
        lineCount += 1
        
print(f"{lineCount} lines were importeted")

# Data Preprocessing already complete.


# Feature Extraction
# This will turn the words into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(words)

# Splitting Datasets
# This splits the dataset into testing and training datasets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model Training
# Train the model using training dataset
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Model Evaluation
# Evaluate the model performance against testing dataset
accuracy = naive_bayes.score(X_test, y_test)
print(f"Accuracy: {accuracy}")