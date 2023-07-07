# import libraries
import string
import pandas as pd
import numpy as np
import sklearn
import math
import re
import matplotlib.pyplot as plt
import seaborn as sns


# Import necessary modules from sklearn
from sklearn.feature_extraction.text import TfidfVectorizer as vectorize
from sklearn.model_selection import train_test_split
from sklearn import pipeline
from sklearn import linear_model
from sklearn import metrics

# create dataframe with pandas
df = pd.read_csv('Language Detection.csv')

# Global Variables
cols = df.columns.tolist()
textCol = df[cols[0]]
langCol = df[cols[1]]
#gramType = 2
MasterLibrary = [] # length is number of unique words

textTrain, textTest, langTrain, langTest = train_test_split(textCol, langCol, test_size=0.2)

corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]

corpus = textTrain.copy()
corpusLength = len(corpus) # number of documents in corpus

# region Testing cleanData(input)

# testLine = textCol[0]
# print(f"Pre-Formatting: \n {testLine}")
# print(f"Post-Formatting: \n {cleanData(testLine)}")
# endregion

# region NOTES on TF IDF

# ==============================================Term Frequency Component==============================================

# use vectorization to turn text into numbers so teh model can understand it
# Create a "libraryVector" whose features consist of parts of a input document. 
#       They can be either 1 word, 2 words or 3 words per feature (unigram, bigram, trigram).
# Create a "documentVector" that is parallel to the "libraryVector".
#       The corresponding position in this vector shows the numbner of times that the feature appears in the document.
# Scan the text 1 feature at a time.
# Find the feature in the "libraryVector" and add 1 to the corresponding index in the "documentVector"

# You can divide the "documentVector" by its length to get a value between 0 and 1

# ==============================================Word Weighting Component==============================================

# The inverse document frequency (IDF) is how many times a feature appears in all of the documents.
# IDF(variable) = log(total number of documents / number of documents containing 'variable' )

# endregion

# region            cleanData(input [String])
# Function:         Return a string that has had all punctuation removed and been set to lowercase
# Sample Input:     cleanData("This, is a string!& of text")
# Sample Output:    this is a string of text
# Purpose:          Used to clean input documents for pre-processing
# endregion
def cleanData(input):
    for char in string.punctuation:
        input = input.replace(char, "")
    input = input.lower()
    return input

# region            gramSplit(doc [String], gramType [int])
# Function:         Return a list (vector) of either unigram or bigram features from an input string. 
#                   gramType determines if unigram or bigram. unigram --> bigram --> 2.
# Sample Input:     gramSplit("This is a test string", 2)
# Sample Output:    ['This is', 'a test', 'string']
# Purpose:          Used to vectorize text for natural language processing
# endregion
def gramSplit(doc, gramType):

    # Null document input edge case
    if doc == None:
        print("inputted document is NULL, exiting")
        return
    
    # split input into individual words and create output list
    uniFeatureList = doc.split()
    libraryVector = []
    n = 0

    if gramType == 2:
        
        documentLength = math.floor(len(uniFeatureList)/2)

        # Since it is a bigram, create pairs of features and append to output list
        for count in range(documentLength):
            feature = uniFeatureList[n] + " " + uniFeatureList[n+1]
            libraryVector.append(feature)
            n = n + 2

        # If input has odd number of words, add last word to end of output list
        if (len(uniFeatureList) % 2) == 1:
            libraryVector.append(uniFeatureList[-1])

    elif gramType == 1:
        
        documentLength = len(uniFeatureList)
        
        for feature in uniFeatureList:
            libraryVector.append(feature)
    
    # Bad gramType input edge case 
    else:
        print(f"{gramType} is not a valid input, please input an integer 1 or 2")
        return

    return libraryVector

# region            removeDupes()
# Function:         remvoes duplicate features from master list (MasterLibrary)
# endregion
def removeDupes():
    # specify list is same as global list declare
    global MasterLibrary
    MasterLibrary = list(set(MasterLibrary))

# region            appendMaster(document [String])
# Function:         Appends the master list (vector) if new passage is added.
# Sample Input:     
#                   MasterLibrary = ['This is', 'a string', 'of text']
#                   appendMaster("This is a new string of text")
# Sample Output:    ['This is', 'a string', 'of text', 'a new', 'string of', 'text']
# Purpose:          When new documents are added, they get a new spot
# endregion
def appendMaster(document, gramType):
    inDoc = gramSplit(cleanData(document), gramType)
    for feature in inDoc:
        MasterLibrary.append(feature)
    removeDupes()
    #print(f"Successfully added '{document}' to the MasterLibrary")

# region            TermFrequency(document [String])
# Function:         Calculates how often a term appears in a given document, based on the Master Library
# Sample Input:     
#                   MasterLibrary = ['This is', 'a string', 'of text']
#                   appendMaster("This is a new string of text")
# Sample Output:    ['This is', 'a string', 'of text', 'a new', 'string of', 'text']
# Purpose:          When new documents are added, they get a new spot
# endregion
def termFrequency(document, gramType):
    
    #print(f"MasterLibrary should be populated: {MasterLibrary}\nCalculating term frequency for document")
    
    # split terms of input phrase into correct vector. Determine length of output vector.
    sp = gramSplit(cleanData(document), gramType)
    #print(f"the document has length {len(sp)} --> {sp}")
    documentVector = np.zeros(len(MasterLibrary), dtype=int)
    
    # go through each feature from the input document. If it exists in the Mastery Library, 
    # find at which index and add 1 to the corresponding index in teh documentVector.
    
    for feature in sp:
        if feature in MasterLibrary:
            index = MasterLibrary.index(feature)
            documentVector[index] += 1
            #print(documentVector)
        else:
            # The code has found a feature that is no in the MasterLibrary, add it and try again.
            print(f"Element '{feature}' not found in MasterLibrary, appending MasterLibrary")
            appendMaster(document, gramType)
            documentVector = np.concatenate((documentVector, np.zeros(1, dtype=int)), axis=0)
            if feature in MasterLibrary:
                index = MasterLibrary.index(feature)
                documentVector[index] += 1
                #print(documentVector)
            else:
                # Panic, something is wrong
                print("PANIC")
                
    #print(documentVector)
    
    return documentVector

# region            IDT(document [list])
# Function:         
# Purpose:          
# endregion
def IDT(document):
    
    for aM in range(corpusLength):
        line = document[aM]
        appendMaster(line, 1)
    
    #print(MasterLibrary)
    
    TF = []
    
    for tf in range(corpusLength):
        tempTF = termFrequency(document[tf], 1)
        TF.append(np.array(tempTF)) 

    idfARRAY = np.zeros(len(MasterLibrary))
    
    
    for feature in MasterLibrary:                                       # for each feature in the library
        k = 0                                                           # by default none of the documents have the feature
        idf = 0
        index = 0
        for i in range(corpusLength):                                   # look thrugh each document in the corpus
            index = MasterLibrary.index(feature)                        # get the index of that feature from the MasterLibrary
            #print(f"in document {i}, {feature} appears {TF[i][index]} time")
            if TF[i][index] > 0:                                        # if the value of the i-th document for that index is greater than 0
                k += 1                                                  # add 1 to the number of documents that feature appears in
            else:
                continue
        #print(k)
        idf = np.log10(corpusLength/k)                               # once you have finished countering how many documents the feature appears in (k value) calculate the IDF for that feature
        idfARRAY[index] = idf
        
    return idfARRAY, TF

# region            tf_idf(document [list])
# Function:         
# Purpose:          
# endregion
def tf_idf(document):
    
    row = 0
    tfIDF = []
    idf, TF = IDT(document)
    
    for row in range(corpusLength):
        
        line = document[row]
        temp = gramSplit(cleanData(line), 1)
        docLength = len(temp)
        
        termfreq = TF[row]/docLength
        
        for feature in MasterLibrary:
            index = MasterLibrary.index(feature)
            #print(f"{feature:>25}: {idf[index]:>20}      x {termfreq[index]:>10}     = {(idf[index]*termfreq[index]):>1}")
        
        final = idf * termfreq
        tfIDF.append(final)
        print(f"{final}\n")
    return tfIDF

termFrequency(textCol, 1)


#vEkin = tf_idf(corpus)

# region
#vect = vectorize(ngram_range=(1,2), analyzer='char')
#pipe = pipeline.Pipeline([('vec', vect), ('clf',linear_model.LogisticRegression())])
#out = pipe.fit(textTrain, langTrain)
# print(f"\n\n\n{out}")
#predictedValues = pipe.predict(textTest)
# accuracy = metrics.accuracy_score(langTest, predictedValues)*100
# print("%.2f" % round(accuracy, 2)+"%")
# endregion
