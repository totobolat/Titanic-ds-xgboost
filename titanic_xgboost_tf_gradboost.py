import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import tensorflow_decision_forests as tfdf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('/kaggle/input/titanic/train.csv',sep=',')
pd.set_option('display.max_columns',None)
df_train.head()


sns.countplot(x = df_train['Survived'], color= 'green')

pd.isna(df_train).sum()

df_train['Cabin'].describe()

df_train['Embarked'].describe()

df_train['Age'].describe()

df_train_feat = df_train.iloc[:, [2,4,5,6,7,9,10,11]]
df_train_label = df_train.iloc[:, [1]]
df_train_feat.head()

df_train_enc = df_train['Sex']
df_train_enc_emb = df_train['Embarked']
df_train.drop('Sex',axis=1,inplace=True)
df_train.drop('Embarked',axis=1,inplace=True)
df_train.head()

df_train_enc = df_train_enc.to_frame()
lb = LabelEncoder()
df_train_enc['Sex'] = lb.fit_transform(df_train_feat['Sex'].values)
df_train_enc['Embarked'] = lb.fit_transform(df_train_feat['Embarked'].values)
df_train_enc

df_train = pd.concat([df_train,df_train_enc],axis=1)
df_train.head()

df_train['Age']=df_train['Age'].fillna(df_train['Age'].mean())
pd.isna(df_train).sum()

df_train.drop(['Cabin'],axis=1,inplace=True)
df_train.drop("Name",axis=1,inplace=True)
df_train.drop("PassengerId",axis=1,inplace=True)
df_train.drop("Ticket",axis=1,inplace=True)
df_train.head()
#df_train_for_age = df_train.dropna(subset=['Age'],inplace=False).copy()
#y_age = df_train_for_age['Age']
#df_train_for_age.drop(['Age'],axis=1,inplace=True)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="Survived")
model = tfdf.keras.GradientBoostedTreesModel(
    num_trees=1000,
    growing_strategy="BEST_FIRST_GLOBAL",
    max_depth=100,
    split_axis="SPARSE_OBLIQUE",
    )
model.fit(train_ds)

model.compile(metrics=["accuracy"])
print(model.evaluate(train_ds))
#model.fit(train_dataset1)
#model.compile(metrics=["accuracy"])
#print(model.evaluate(test_dataset1))

xg_df_train = df_train.copy()
xg_df_train_y = xg_df_train['Survived']
xg_df_train.drop('Survived',axis=1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(xg_df_train, xg_df_train_y, test_size=0.1, random_state=0)
evalset = [(X_test, y_test)]
model_xgb = XGBClassifier(n_estimators=1000,learning_rate=0.01,max_depth=6,subsample = 0.8,min_child_weight=0.2,booster='gbtree'
                          ,tree_method='hist',max_delta_step=1,objective='binary:hinge',colsample_bytree=0.8,gamma = 10,reg_alpha=0.005)
model_xgb.fit(xg_df_train, xg_df_train_y)
#model_xgb.fit(X_train, y_train,eval_set=evalset)
#y_pred = model_xgb.predict(X_test)
#predictions = [round(value) for value in y_pred]
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_test.head()

df_test.drop('Cabin',axis=1,inplace=True)
df_test.drop("Name",axis=1,inplace=True)
df_test_pid = df_test['PassengerId']
df_test.drop("PassengerId",axis=1,inplace=True)
df_test.drop("Ticket",axis=1,inplace=True)
df_test.head()

df_test_enc = df_test['Sex']
df_test_enc_emb = df_test['Embarked']
df_test_enc = df_test_enc.to_frame()
lb = LabelEncoder()
df_test_enc['Sex'] = lb.fit_transform(df_test['Sex'].values)
df_test_enc['Embarked'] = lb.fit_transform(df_test['Embarked'].values)
df_test_enc

df_test.drop('Sex',axis=1,inplace=True)
df_test.drop('Embarked',axis=1,inplace=True)

df_test = pd.concat([df_test,df_test_enc],axis=1)
df_test.head()

df_test_gb = tfdf.keras.pd_dataframe_to_tf_dataset(df_test)

model.predict(df_test_gb, verbose=0)

predictions_df_test = model.predict(df_test_gb, verbose=2)
predictions_df_test = (predictions_df_test > 0.5).flatten()
print(predictions_df_test)

submit_res = pd.DataFrame(columns=['Survived'],data=predictions_df_test)
submit_res.head()

#submit_res_enc = submit_res.to_frame()
submit_res_enc = submit_res.copy()
lb = LabelEncoder()
submit_res_enc['Survived'] = lb.fit_transform(submit_res['Survived'].values)
submit_res_enc

df_test_pid = df_test_pid.to_frame()
df_test_pid

submit_res_final = pd.concat([df_test_pid,submit_res_enc],axis=1)
submit_res_final

submit_res_final.to_csv('submit_res_final.csv', index = False)

predictions_df_test_xg = model_xgb.predict(df_test)
predictions_df_test_xg = (predictions_df_test_xg > 0.5).flatten()
print(predictions_df_test_xg)

submit_res_xg = pd.DataFrame(columns=['Survived'],data=predictions_df_test_xg)
submit_res_xg_enc = submit_res_xg.copy()
lb = LabelEncoder()
submit_res_xg_enc['Survived'] = lb.fit_transform(submit_res_xg['Survived'].values)
submit_res_xg_final = pd.concat([df_test_pid,submit_res_xg_enc],axis=1)
submit_res_xg_final

submit_res_xg_final.to_csv('submit_res_xg_final_tweaked.csv', index = False)

#Submission result is 0.78708