import NLP_Classes as nlp
import pandas as pd
import sklearn as sk

df = pd.read_csv('testVectorization.csv')
cols = df.columns.tolist()
textCol = df[cols[0]]                                          # List of contents in Text column
langCol = df[cols[1]]   

# Create corpus object
corpus = nlp.Corpus(df, textCol, langCol, 1)

# Split corpus into training and testing datasets
textTrain, textTest, langTrain, langTest = corpus.splitCorpus()

# Create Vectorize object
vec = nlp.Vectorize()

# Vectorize the text using TF IDF teqchnizue
tfIDF = vec.tf_idf(textCol)

# create a pipeline to handle logistic regression
pipeline = nlp.Pipeline()

out = pipeline.addStep([('vec', tfIDF), ('char', LogisticRegression())])





#textTrain, textTest, langTrain, langTest = train_test_split(textCol, langCol, test_size=0.2)


#corpus = textTrain.copy()
#corpusLength = len(corpus)