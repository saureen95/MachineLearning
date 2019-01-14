import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
#reading data 
data = pd.read_csv('data.csv', encoding = "ISO-8859-1")
data.head(1)
#testing and training data splitting
train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']
# Removing punctuations
dataSlice= train.iloc[:,2:27]
dataSlice.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)
#renaming columns
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
dataSlice.columns= new_Index
dataSlice.head(5)
#converting headlines to lowercase 
for index in new_Index:
    dataSlice[index]=dataSlice[index].str.lower()
dataSlice.head(1)
headlines = []
for row in range(0,len(dataSlice.index)):
    headlines.append(' '.join(str(x) for x in dataSlice.iloc[row,0:25]))


# N grams
basicvectorizer = CountVectorizer(ngram_range=(1,1))
basictrain = basicvectorizer.fit_transform(headlines)
print(basictrain.shape)
basicmodel = GaussianNB()
basicmodel = basicmodel.fit(basictrain.toarray(), train["Label"])
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest.toarray())
pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
#printing accuracies
print (classification_report(test["Label"], predictions))
#same for N gram(1,2)
basicvectorizer2 = CountVectorizer(ngram_range=(1,2))
basictrain2 = basicvectorizer2.fit_transform(headlines)
print(basictrain2.shape)

basicmodel2 = GaussianNB()
basicmodel2 = basicmodel2.fit(basictrain2, train["Label"])

basictest2 = basicvectorizer2.transform(testheadlines)
predictions2 = basicmodel2.predict(basictest2)

pd.crosstab(test["Label"], predictions2, rownames=["Actual"], colnames=["Predicted"])

print (classification_report(test["Label"], predictions2))
print (accuracy_score(test["Label"], predictions2))
print (classification_report(test["Label"], predictions2))
print (accuracy_score(test["Label"], predictions2))

#same for N gram(2,3)
basicvectorizer3 = CountVectorizer(ngram_range=(2,3))
basictrain3 = basicvectorizer3.fit_transform(headlines)
print(basictrain3.shape)

basicmodel3 = GaussianNB()
basicmodel3 = basicmodel3.fit(basictrain3, train["Label"])

basictest3 = basicvectorizer3.transform(testheadlines)
predictions3 = basicmodel3.predict(basictest3)

pd.crosstab(test["Label"], predictions3, rownames=["Actual"], colnames=["Predicted"])

print (classification_report(test["Label"], predictions3))
print (accuracy_score(test["Label"], predictions3)) 
print (accuracy_score(test["Label"], predictions))
