import string
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import math
import logging

from sklearn import pipeline
from sklearn import linear_model
from sklearn import metrics

# create a corpus class.
class Corpus():
    
    MasterLibrary = []
    libraryVector = []
    textTrain = [3, 2, 1]    # need to automate the length of master library for right after read in.
    gramType = 0
    
    # Each corpus gets a dataframe, list of columns in dataset, specification on unigram or bigram analysis for vectorizationm and a list of unique features
    def __init__(self, df, textCol, langCol, gT):
        self.df = df                                                        # Pandas Dataframe
        
        self.textCol = textCol                                              # text Column
        self.langCol = langCol                                              # Language Column
        Corpus.gramType = gT                                                # Set gramType attribute
        self.MasterLibrary = []                                             # List of unique features in corpus
    
    def cleanData(input):                                                   # Method to clean inputed data from the corpus. Removes unecessary characters (punctuation) and uppercase letters.
        for char in string.punctuation:
            input = input.replace(char, "")
        input = input.lower()
        return input
    
    def gramSplit(document: string):
        uniFeatureList = document.split()
        n = 0
    
        if Corpus.gramType == 2:                                              # Bigram
            documentLength = math.floor(len(uniFeatureList) / 2)
            
            if len(uniFeatureList) % 2 == 1:                                # If odd number of features, reduce range by 1 and add in the last feature as a unigram. Order in list does not matter.
                documentLength = documentLength - 1
                Corpus.libraryVector.append(uniFeatureList[-1]+ "|")               # '|' delimeter
            
            for n in range(documentLength):
                feature = uniFeatureList[n] + "|" + uniFeatureList[n+1]     # '|' delimeter
                Corpus.libraryVector.append(feature)
                n = n + 2
                
        elif Corpus.gramType == 1:                                            # Unigram
            documentLength = len(uniFeatureList)
            for feature in uniFeatureList:
                Corpus.libraryVector.append(feature)
                
        else:
            print(f"{Corpus.gramType} is not a valid input, must be an integer with the value of either 1 or 2")
            return
        
        return
                                                                            
    def removeDupes():                                                  # Remove dulplicate entries in MasterLibrary
        Corpus.MasterLibrary = list(set(Corpus.MasterLibrary))
    
    def appendMaster(document):                                       # Add Set of entries into MasterLibrary
        for feature in Corpus.libraryVector:
            Corpus.MasterLibrary.append(feature)
        Corpus.removeDupes()
                                                                            
    def splitCorpus(self):                                                  # Split corpus into training and testing datasets
        textTrain, textTest, langTrain, langTest = train_test_split(self.textCol, self.langCol, test_size=0.2)
        return textTrain, textTest, langTrain, langTest

    def getTrainingLength(self):
        return len(self.textTrain)
    
    def getMasterLength(self):
        return len(self.MasterLibrary)
    
class Vectorize():
    
    corpusLength = 0
    gramType = 0
    
    def __init__(self):
        Vectorize.corpusLength = Corpus.getTrainingLength(Corpus)
        Vectorize.gramType = Corpus.gramType
    
    def termFrequency(document):
        documentVector = np.zeros(len(Corpus.MasterLibrary), dtype=int)
        
        #print(Corpus.MasterLibrary)
        for feature in Corpus.libraryVector:                                                  # Count feature frequency if it exists, if it does not, add it.
            if feature in Corpus.MasterLibrary:
                index = Corpus.MasterLibrary.index(feature)
                documentVector[index] += 1
                
            else:                                                           # Feature is not in the MasterLibrary, add it and try again.
                logging.basicConfig(level=logging.INFO)
                logging.info(f"Element '{feature}' not found in MasterLibrary, appending MasterLibrary")
                
                Corpus.appendMaster(document, Corpus.gramType)
                documentVector = np.concatenate((documentVector, np.zeros(1, dtype=int)), axis=0)
                
                if feature in Corpus.MasterLibrary:
                    index = Corpus.MasterLibrary.index(feature)
                    documentVector[index] += 1

                else:
                    logging.basicConfig(level=logging.DEBUG)
                    logging.debug(f"Code was unable to add: '{document}'")
                    documentVector[index] += 1
        
        return documentVector
    
    def IDT(self, document):
        TF = []
        idfARRAY = np.zeros(Corpus.getMasterLength(Corpus))
        
        for i in range(self.corpusLength):
            Corpus.gramSplit(Corpus.cleanData(document[i]))
            Corpus.appendMaster(Corpus.libraryVector)
        
        print(Corpus.MasterLibrary)
        
        for j in range(self.corpusLength):
            TF.append(np.array(Vectorize.termFrequency(document[i])))
        
        #print(TF)
        
        for feature in Corpus.MasterLibrary:                                  # for each feature in the library
            k = 0                                                           # by default none of the documents have the feature
            idf = 0
            index = 0
            for i in range(self.corpusLength):                              # look thrugh each document in the corpus
                index = Corpus.MasterLibrary.index(feature)                   # get the index of that feature from the MasterLibrary
                if TF[i][index] > 0:                                        # if the value of the i-th document for that index is greater than 0
                    k += 1                                                  # add 1 to the number of documents that feature appears in
                else:
                    continue
            idf = np.log10(self.corpusLength/k)                               # once you have finished countering how many documents the feature appears in (k value) calculate the IDF for that feature
            idfARRAY[index] = idf
            
        return idfARRAY, TF

    def tf_idf(self, document):
        row = 0
        tfIDF = []
        idf, TF = self.IDT(document)
        
        for row in range(self.corpusLength):
            line = document[row]
            temp = Corpus.gramSplit(Corpus.cleanData(line))
            docLength = len(temp)
            termfreq = TF[row]/docLength
            
            for feature in Corpus.MasterLibrary:
                index = Corpus.MasterLibrary.index(feature)
            
            final = idf * termfreq
            tfIDF.append(final)
            
            logging.basicConfig(level=logging.DEBUG)
            logging.debug(f"{final}\n") 
        return tfIDF
    
    