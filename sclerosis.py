import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import  LazyClassifier

from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.metrics

data=pd.read_csv('conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv')
print(data.columns)
print(data.info())
print(data.describe())
print(data.isna().sum())

'''for i in data.columns.values:
    sn.boxplot(data[i])
     plt.show()'''

'''for i in data.select_dtypes(include='number').columns.values:
    if len(data[i].value_counts()) <=5:
        val=data[i].value_counts().values
        index=data[i].value_counts().index
        plt.pie(val,labels=index,autopct='%1.1f%%')
        plt.title(f'The PIE Chart information of {i} column')
        plt.show()


for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

x_val=data.drop('group',axis=1)
y_val=data['group']

for i in data.columns.values:
    if data[i].isna().sum() > 0:
        if data[i].skew() > 0:
            data[i]=data[i].fillna(data[i].mean())
        elif data[i].skew() < 0:
            data[i]=data[i].fillna(data[i].median())

'''sn.scatterplot(x_val.head(100))
sn.scatterplot(y_val.head(100))
plt.title('Scatter plot of all the column with group column')
plt.show()'''

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()'''

print(data['Final_EDSS'].skew())
print(data['Initial_EDSS'].skew())
print(data['Schooling'].skew())

print(data.isna().sum())

print(data['group'].value_counts())

data['group']=data['group'].replace([2,1],[0,1])
x=data[['Unnamed: 0','Gender','Breastfeeding','Varicella','Initial_EDSS','Final_EDSS']]
y=data['group']

x_train,x_test,y_train,y_test=train_test_split(x,y)

lr = LogisticRegression(max_iter=20)
lr.fit(x_train, y_train)
pred=lr.predict(x_test)
print('The logistic regression: ', lr.score(x_test, y_test))

lazy=LazyClassifier(verbose=1)
models,predict=lazy.fit(x_train,x_test,y_train,y_test)
print(models)


Y=pd.get_dummies(y)
x_tran,x_tst,y_tran,y_tst=train_test_split(x,Y)
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.tanh))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
hist=models.fit(x_tran,y_tran,batch_size=20,epochs=65)

plt.plot(hist.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(hist.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam optimizer')
plt.legend()
plt.show()


models1=Sequential()
models1.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models1.compile(optimizer='rmsprop',loss=keras.losses.binary_crossentropy,metrics='accuracy')
histo=models1.fit(x_tran,y_tran,batch_size=20,epochs=65)

plt.plot(histo.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(histo.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam rmsprop')
plt.legend()
plt.show()