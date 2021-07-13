from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
root = os.path.dirname(__file__)
path_df = os.path.join(root, 'dataset/Chronic_final.csv')
data = pd.read_csv(path_df)
cat_cols = ['Age','BP','specific_gravity','Albumin','sugar','Blood_Gluc_rand','Blood_Urea','Serum_Cr','sodium','potassium','hemoglobin',
        'packed_cell_volume','wbc_cnt','htn','diabetes','CAD','apetite','pedal_edema']
X = data.iloc[:, :-1].values
y = data.iloc[:, 18].values
print(X[0:1]) 
#print(data['BP'].head()) 
#print(data['class'].head()) 
Labelx=LabelEncoder()
X[:,13]=Labelx.fit_transform(X[:,13])
X[:,14]=Labelx.fit_transform(X[:,14])
X[:,15]=Labelx.fit_transform(X[:,15])
X[:,16]=Labelx.fit_transform(X[:,16])
X[:,17]=Labelx.fit_transform(X[:,17])


X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.25)
print("fffffff")
print(X_train[0:1]) 
print(Y_train[0:10])
print("test")
print(X_test[0:1])
print(Y_test[0:1])
# We don't scale targets: Y_test, Y_train as SVC returns the class labels not probability values
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

clf = RandomForestClassifier()

# Training the classifier
clf.fit(X_train, Y_train)
#print(X_test[0:1])
#print(clf.predict(X_test[0:1]))


lr=LogisticRegression()
lr.fit(X_train,Y_train)
#y_pred_lr=lr.predict(X_test)

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib

#create a dictionary of base learners
estimators=[('rfc', clf), ('lr', lr)]
#create voting classifier
majority_voting = VotingClassifier(estimators, voting='hard')

#fit model to training data
majority_voting.fit(X_train, Y_train)
#test our model on the test data
majority_voting.score(X_test, Y_test)

# save best model to current working directory
#joblib.dump(majority_voting, "cronicmodel.pkl")

# load from file and predict 
#model_max_v = joblib.load("cronicmodel.pkl" )

# get predictions from best model above
y_preds_mv = majority_voting.predict(X_test)
print('majority voting accuracy: ',majority_voting.score(X_test, Y_test))
print('\n')

import pylab as plt
labels=[0,1]
cmx=confusion_matrix(Y_test,y_preds_mv, labels)
print(cmx)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cmx)
plt.title('Confusion matrix Majority Voting classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('\n')
print(classification_report(Y_test, y_preds_mv))


# Testing model accuracy. Average is taken as test set is very small hence accuracy varies a lot everytime the model is trained
acc = 0
acc_binary = 0
for i in range(0, 20):
    Y_hat = clf.predict(X_test)
    Y_hat_bin = Y_hat>0
    Y_test_bin = Y_test>0
    acc = acc + accuracy_score(Y_hat, Y_test)
    acc_binary = acc_binary +accuracy_score(Y_hat_bin, Y_test_bin)

print("Average test Accuracy:{}".format(acc/20))
print("Average binary accuracy:{}".format(acc_binary/20))

# Saving the trained model for inference
model_path = os.path.join(root, 'dataset/rfc.sav')
joblib.dump(clf, model_path)
model_path1 = os.path.join(root, 'dataset/mv.sav')
joblib.dump(majority_voting, model_path1)
# Saving the scaler object
scaler_path = os.path.join(root, 'dataset/scaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(Labelx, scaler_file)

scaler_path = os.path.join(os.path.dirname(__file__), 'dataset/scaler.pkl')
scaler = None
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

a=48
b=80
c=1.02
d=1
e=0
g=121
h=36	
i=1.2	
j=107.61625	
k=3.60925	
l=15.4	
m=44	
n=7800	
o="yes"	
p="yes"	
q="no"	
r="good"	
s="no"	
t=1
vector = np.vectorize(np.float)
check = np.array([a,b,c,d,e,g,h,i,j,k,l,m,n,o,p,q,r,s]).reshape(1, -1)

Labe=LabelEncoder()
check[:,13]=Labe.fit_transform(check[:,13])
check[:,14]=Labe.fit_transform(check[:,14])
check[:,15]=Labe.fit_transform(check[:,15])
check[:,16]=Labe.fit_transform(check[:,16])
check[:,17]=Labe.fit_transform(check[:,17])
model_path = os.path.join(os.path.dirname(__file__), 'dataset/rfc.sav')




 
check = vector(check)
#print(check) 

print(X_test[0:1])
print(check[[0]] )
clf = joblib.load(model_path)
B_pred = clf.predict(check[[0]])
if B_pred == 1:
    print("cronical kidney disease detected")
if B_pred == 0:
    print("No disease detected")