import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

# Create a dictionary of words in the email including their frequency
def createDictionary(data):
    receievedEmails = [os.path.join(data, x) for x in os.listdir(data)]
    wordList = []
    for email in receivedEmails:
        with open(email) as mail: 
            for i, sentence in enumerate(mail):
                if i == 2:
                    words = sentence.split()
                    wordList += words
    dictionary = Counter(wordList)
    
    extractedList = dictionary.keys()
    
    #Remove single characters and common words that are not 
    # helpful in deciding if the email is spam
    for keys in extractedList:
        if key.isalpha() == False:
            del dictionary[key]
        elif len(key) == 1:
            del dictionary[key]
    dictionary = dictionary.most_common(3000)
    return dictionary

# Extract word count vector of 3000 dimension for each email
# from the training set.
def featureExtraction(emails):
    data = [os.path.join(emails, x) for x in os.listdir(emails)]
    matrix = np.zeros((len(data), 3000))
    x = 0
    for file in data:
        with open(file) as f:
            for i, sentence in enumerate(f):
                if i == 2:
                    words = sentence.split()
                    for eachWord in words:
                        y = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == eachWord:
                                y = i
                                matrix[x, y] = words.count(eachWord)
            x = x + 1
    return matrix

# Train classifiers using scikit-learn machine learning library
directory = 'lingspam_public\\lemm_stop\\train-mails'
dictionary = createDictionary(directory)
label = np.zeros(702)
label[351:701] = 1
matrix = featureExtraction(directory)
firstModel = LinearSVC()
secondModel = MultinomialNB()
firstModel.fit(matrix, label)
secondModel.fit(matrix, label)
directory2 = 'lingspam_public\\lemm_stop\\test-mails'
matrix2 = featureExtraction(directory2)
label2 = np.zeros(260)
label2[130:260] = 1
firstResult = firstModel.predict(matrix2)
secondResult = firstModel.predict(matrix2)
