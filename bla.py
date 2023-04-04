import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics
nltk.download('stopwords')
nltk.download('wordnet')

from langdetect import detect
from googletrans import Translator
from gensim.models import Word2Vec

# Creating DataFrames for training and testing
trainData = pd.read_csv("data/mediaeval-2015-trainingset.txt", sep="	")
testData = pd.read_csv("data/mediaeval-2015-testset.txt", sep="	")
df_train = pd.DataFrame(data=trainData)
df_test = pd.DataFrame(data=testData)


# Data Characterization
print(trainData.head())
# print(trainData.info())  # Metadata of training data, including size
# print(df_test.shape)  # Size of testing data

df_train.rename(columns = {'imageId(s)':'imgs'}, inplace = True)
imgCount = df_train.groupby(df_train.imgs.str.split('_').str[0])['tweetId'].nunique()
# print (imgCount)

#Removing remaining twitter handles @username
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'@\w*', "", text))

#Removing emojis
emojis = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)

df_train['tweetText'] = df_train['tweetText'].apply(lambda text: emojis.sub(r'', text) if emojis.search(text) else text)

#Cleaning symbols - ampersand and newline
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'&amp;|\\n', '', text))

#Removing urls
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'http\S+', '', text))
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'\\\/\S+', '', text))

#Removing whitespace
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: " ".join(text.split()))

#Changing 'humor' to 'fake'
df_train.loc[(df_train.label == 'humor'),'label'] = 'fake'
df_test.loc[(df_test.label == 'humor'),'label'] = 'fake'

# Initialise stopwords
stopwords = nltk.corpus.stopwords.words()
stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', '...'])

#Removing stopwords
df_train['filteredTweet'] = [' '.join(w for w in tweet.split() if w not in stopwords) for tweet in df_train['tweetText']]

#Lemmatising
tokeniser = nltk.tokenize.WhitespaceTokenizer()
lemmatiser = nltk.stem.WordNetLemmatizer()

df_train['lemmatisedTweet'] = df_train['filteredTweet'].apply(lambda x: ' '.join([lemmatiser.lemmatize(w) for w in tokeniser.tokenize(x)]))
# print(df_train.head(10))

tar_train = df_train.label
ft_train = df_train.lemmatisedTweet
tar_test = df_test.label
ft_test = df_test.tweetText

#Init Bag-of-Words
count_vectoriser = CountVectorizer(stop_words='english')
count_train = count_vectoriser.fit_transform(ft_train)
count_test = count_vectoriser.transform(ft_test)

clf = MultinomialNB()

#Bag-of-Words
# clf.fit(count_train, tar_train)
# pred = clf.predict(count_test)
# score = metrics.accuracy_score(tar_test, pred)
# print("accuracy:   %0.3f" % score)

# #Calculating F1 score
# TP = 0 
# FP = 0
# TN = 0
# FN = 0

# for true, guess in zip(tar_test, pred):
#     if(true == 'fake' and guess == 'fake'):
#         TP = TP + 1
#     if(true == 'real' and guess == 'fake'):
#         FP = FP + 1
#     if(true == 'real' and guess == 'real'):
#         TN = TN + 1
#     if(true == 'fake' and guess == 'real'):
#         FN = FN + 1
        
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)

# f1 = 2 * ((precision * recall) / (precision + recall))
# print("TP: %d FP: %d TN: %d FN: %d" % (TP, FP, TN, FN))
# print("f1: %0.3f" % f1)
