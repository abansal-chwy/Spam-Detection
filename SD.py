import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk 
nltk.download('stopwords')
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df=pd.read_csv("SMSSpamCollection",delimiter="\t",quoting=3,header=None)

# Output printing out first 5 rows

print(df.head())



#Convert the values in the 'label' column to numerical values using map method as follows: {'ham':0, 'spam':1} 
#This maps the 'ham' value to 0 and the 'spam' value to 1.
#df['label'] = df.label.map({'ham':0, 'spam':1})

#Check the shape of the dataset
print(df.shape) #5572,2

#Bag of words
#using count vectorizer - 
#It tokenizes the string(separates the string into individual words) and gives an integer ID to each token.
#It counts the occurrence of each of those tokens.


#Convert all the strings in the documents set to their lower case. 
#Save them into a list called 'lower_case_documents

documents = []
for docs in range(0,len(df)):
    message=re.sub("[^a-zA-Z]"," ",df.iloc[docs,1]) #removed character replaced by space
    message = message.lower()
    message = message.split()
    ps = PorterStemmer()
    message = [word for word in message if word not in set(stopwords.words("english"))]
    message=" ".join(message)
    documents.append(message)
    
#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)#filter non relevant words #keep max 1500
X=cv.fit_transform(documents).toarray()

#Train the the classification model on bag of words

y=df.iloc[:,0].values # assign dependent variable
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy = 97%
