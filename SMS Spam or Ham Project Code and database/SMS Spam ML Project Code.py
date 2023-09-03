#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Panda and NumPy

import numpy as np
import pandas as pd


# In[ ]:


# Importing Dataset

df = pd.read_csv(r"/content/drive/MyDrive/spam.csv",encoding = "ISO-8859-1")


# In[ ]:


df.sample(5)


# In[ ]:


df.shape


# In[ ]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement


# In[ ]:


# Data Cleaning Process


# In[ ]:


df.info()


# In[ ]:


# Drop last 3 cols

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.head()


# In[ ]:


# Renaming the columns

df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[ ]:


# Giving numerical value to categorical data

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])


# In[ ]:


df.head()


# In[ ]:


# Checking for missing values

df.isnull().sum()


# In[ ]:


# Check for duplicate values

df.duplicated().sum()


# In[ ]:


# Remove duplicates

df = df.drop_duplicates(keep='first')


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


# Exploratory Data Analysis


# In[ ]:


df.head()


# In[ ]:


# Finding Number of ham and spam in target

df['target'].value_counts()


# In[ ]:


#Plotting Pie chart
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[ ]:


# Data is imbalanced

import nltk
nltk.download('punkt')


# In[ ]:


# Num of characters

df['num_characters'] = df['text'].apply(len)
df.head()


# In[ ]:


# Num of words

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df.head()


# In[ ]:


# Num of sentences

df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()


# In[ ]:


df[['target','num_characters','num_words','num_sentences']].describe()


# In[ ]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[ ]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[ ]:


import seaborn as sns

# Plotting Histplot

sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='green')


# In[ ]:


# Plotting Pairplot

sns.pairplot(df,hue='target')


# In[ ]:


# Plotting heatmap

sns.heatmap(df.corr(),annot=True)


# In[ ]:


# Data Preprocessing

# Lower case
# Tokenization
# Removing special characters
# Removing stop words and punctuation
# Stemming


# In[ ]:


from nltk.corpus import stopwords
import string
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)


# In[ ]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[ ]:


df['text'][10]


# In[ ]:


df['transformed_text'] = df['text'].apply(transform_text)
df.head()


# In[ ]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[ ]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[ ]:


df.head()


# In[ ]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[ ]:


len(spam_corpus)


# In[ ]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[ ]:


import numpy as np
a=np.unique(ham_corpus)
len(a)


# In[ ]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Text Vectorization
# using Bag of Words
df.head()


# In[ ]:


# Model Building


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer()


# In[ ]:


# Defining X
X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[ ]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[ ]:


# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[ ]:


X.shape


# In[ ]:


# Defining y
y = df['target'].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[ ]:


def perform(y_pred):
  print("Accuracy : ", accuracy_score(y_test, y_pred))
  print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
  print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
  print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))


# In[ ]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[ ]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
perform(y_pred1)


# In[ ]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
perform(y_pred2)


# In[ ]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
perform(y_pred3)


# In[ ]:


# tfidf --> BNB
# Applying Classification Techniques


# In[ ]:


from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


svc = SVC(kernel='sigmoid', gamma='scale')
bnb = BernoulliNB()
lrc = LogisticRegression(solver='liblinear', penalty='l1')
knc = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators=50, random_state=2)


# In[ ]:


clfs = {
    'Support Vector Machine' : svc,
    'Naive bayes ': bnb,
    'Linear Regression': lrc,
    'K-Nearest Neighbour' : knc,
    'Random Forest': rfc
}


# In[ ]:


def score_find(y_test,y_pred):
    Accuracy = accuracy_score(y_test,y_pred)
    Precision = precision_score(y_test,y_pred)
    Recall = recall_score(y_test, y_pred, average = 'micro')
    F1_score = f1_score(y_test, y_pred, average = 'micro')


    return Accuracy,Precision,Recall,F1_score


# In[ ]:


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


# In[ ]:


print("For Support Vector Machine")
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print()
print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
print(classification_report(y_test, y_pred))
print("**"*27+"\n")

current_accuracy,current_precision,current_recall,current_F1 = score_find(y_test,y_pred)

print("Accuracy - ",current_accuracy)
print("Precision - ",current_precision)
print("Recall - ",current_recall)
print("F1_score - ",current_F1)
print()

cm1 = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['ham', 'spam'])
cm1.plot()

accuracy_scores.append(current_accuracy)
precision_scores.append(current_precision)
recall_scores.append(current_recall)
f1_scores.append(current_F1)


# In[ ]:


print("For Bernoulli Naive Bayes")
bnb.fit(X_train,y_train)
y_pred = bnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print()
print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
print(classification_report(y_test, y_pred))
print("**"*27+"\n")

current_accuracy,current_precision,current_recall,current_F1 = score_find(y_test,y_pred)

print("Accuracy - ",current_accuracy)
print("Precision - ",current_precision)
print("Recall - ",current_recall)
print("F1_score - ",current_F1)
print()

cm1 = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['ham', 'spam'])
cm1.plot()

accuracy_scores.append(current_accuracy)
precision_scores.append(current_precision)
recall_scores.append(current_recall)
f1_scores.append(current_F1)


# In[ ]:


print("For Logistic Regression")
lrc.fit(X_train,y_train)
y_pred = lrc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print()
print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
print(classification_report(y_test, y_pred))
print("**"*27+"\n")

current_accuracy,current_precision,current_recall,current_F1 = score_find(y_test,y_pred)

print("Accuracy - ",current_accuracy)
print("Precision - ",current_precision)
print("Recall - ",current_recall)
print("F1_score - ",current_F1)
print()

cm1 = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['ham', 'spam'])
cm1.plot()

accuracy_scores.append(current_accuracy)
precision_scores.append(current_precision)
recall_scores.append(current_recall)
f1_scores.append(current_F1)


# In[ ]:


print("For K-Nearest Neighbour")
knc.fit(X_train,y_train)
y_pred = knc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print()
print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
print(classification_report(y_test, y_pred))
print("**"*27+"\n")

current_accuracy,current_precision,current_recall,current_F1 = score_find(y_test,y_pred)

print("Accuracy - ",current_accuracy)
print("Precision - ",current_precision)
print("Recall - ",current_recall)
print("F1_score - ",current_F1)
print()

cm1 = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['ham', 'spam'])
cm1.plot()

accuracy_scores.append(current_accuracy)
precision_scores.append(current_precision)
recall_scores.append(current_recall)
f1_scores.append(current_F1)


# In[ ]:


print("For Random Forest")
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print()
print("**"*27 + "\n" + " "* 16 + "Classification Report\n" + "**"*27)
print(classification_report(y_test, y_pred))
print("**"*27+"\n")

current_accuracy,current_precision,current_recall,current_F1 = score_find(y_test,y_pred)

print("Accuracy - ",current_accuracy)
print("Precision - ",current_precision)
print("Recall - ",current_recall)
print("F1_score - ",current_F1)
print()

cm1 = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=['ham', 'spam'])
cm1.plot()

accuracy_scores.append(current_accuracy)
precision_scores.append(current_precision)
recall_scores.append(current_recall)
f1_scores.append(current_F1)


# In[ ]:


print(accuracy_scores)
print(precision_scores)
print(recall_scores)
print(f1_scores)


# In[ ]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores,
                               'Recall':recall_scores,'F1_Score':f1_scores}).sort_values('Precision',ascending=True)
performance_df


# In[ ]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1


# In[ ]:


sns.catplot(x = 'Algorithm', y='value',
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Model improvement
# 1. Change the max_features parameter of TfIdf to 3000 and run code again and store scores in temp_df.


# In[ ]:


# temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores,
#                    'Recall_max_ft_3000':recall_scores,'F1_score_max_ft_3000':f1_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[ ]:


# new_df = performance_df.merge(temp_df,on='Algorithm')


# In[ ]:


# temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores,
#                   'Recall_scaling':recall_scores,'F1_score_scaling':f1_scores}).sort_values('Precision_scaling',ascending=False)


# In[ ]:


# new_df= new_df.merge(temp_df,on='Algorithm')


# In[ ]:


# temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores,
#                  'Recall_num_chars':recall_scores,'F1_score_num_chars':f1_scores}).sort_values('Precision_scaling',ascending=False)


# In[ ]:


# new_df_final=new_df.merge(temp_df,on='Algorithm')


# In[ ]:


# new_df_final


# In[ ]:




