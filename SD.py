import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
with open('SMSSpamCollection','r') as f:
   
    
    df =pd.read_table(f, sep='\t',names=['label','sms_message'],
                  header=None,
                  lineterminator='\n')
# Output printing out first 5 rows
df.to_csv("test.csv")
print(df.head())

#Convert the values in the 'label' column to numerical values using map method as follows: {'ham':0, 'spam':1} 
#This maps the 'ham' value to 0 and the 'spam' value to 1.
df['label'] = df.label.map({'ham':0, 'spam':1})

#Check the shape of the dataset
print(df.shape) #5572,2

#Bag of words
#using count vectorizer - 
#It tokenizes the string(separates the string into individual words) and gives an integer ID to each token.
#It counts the occurrence of each of those tokens.


#Convert all the strings in the documents set to their lower case. 
#Save them into a list called 'lower_case_documents

documents = []
for docs in df['sms_message']:
    documents.append(docs)
print (documents)

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

#Remove all punctuation from the strings in the document set. 
#Save them into a list called 'sans_punctuation_documents'.

sans_punctuation_documents = []
import string

for i in lower_case_documents:
    for c in string.punctuation:
        i=i.replace(c,"")
    sans_punctuation_documents.append(i)    
    
print(sans_punctuation_documents)


#Tokenization
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(" "))
    
print(preprocessed_documents)    

#Count Frequencies
frequency_list = []
import pprint
from collections import Counter
feq={}

for i in preprocessed_documents:
    for j in i:
        if j  in feq:
            feq[j]+=1
        else:
            feq[j]=1
for key in feq:
    frequency_list.append(feq[key])
pprint.pprint(frequency_list)

#Splitting into training and testing set
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

