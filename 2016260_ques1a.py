import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle
from mlxtend.data import loadlocal_mnist


X_train, y_train= loadlocal_mnist(images_path='./train-images.idx3-ubyte', labels_path='./train-labels.idx1-ubyte')
X_test, y_test= loadlocal_mnist(images_path='./t10k-images.idx3-ubyte', labels_path='./t10k-labels.idx1-ubyte')#fine
#print (X_train.shape, y_train.shape, y_test.shape, X_test.shape)

# LIMIT= 127
# for i in range(len(X_train)):
#     for j in range(len(X_train[0])):
#         if(X_train[i][j]<= LIMIT):
#             X_train[i][j]= 0
#         else:
#             X_train[i][j]= 1
#
# for i in range(len(X_test)):
#     for j in range(len(X_test[0])):
#         if(X_test[i][j]<= LIMIT):
#             X_test[i][j]= 0
#         else:
#             X_test[i][j]= 1


# x_train= []
# x_test= []
Y_train= []
Y_test= []

for i in range(len(y_train)):
    if(y_train[i]== 1 or y_train[i]== 2):
        #x_train.append(X_train[i])
        Y_train.append(y_train[i]-1)


for i in range(len(y_test)):
    if(y_test[i]== 1 or y_test[i]== 2):
        #x_test.append(X_test[i])
        Y_test.append(y_test[i]-1)


y_train= Y_train
y_test= Y_test#fine
#print (len(x_train), len(y_train))
#print (len(x_test), len(y_test))


# with open('./bintrain.pkl', 'wb+') as f:
#     pickle.dump(x_train, f)
# with open('./bintest.pkl', 'wb+') as f:
#     pickle.dump(x_test, f)
f= open('./bintrain.pkl', 'rb')
x_train= pickle.load(f)
f= open('./bintest.pkl', 'rb')
x_test= pickle.load(f)

#fine
# counttrain0= np.ones(shape= (784, 2))
# counttrain1= np.ones(shape= (784, 2))
# counttrain0= counttrain0*(1.0/100.0)
# counttrain1= counttrain1*(1.0/100.0)
# for i in range(len(counttrain0)):
#     for k in range(len(x_train)):
#         if(x_train[k][i]== 0 and y_train[k]== 0):
#             counttrain0[i][0]+= 1
#         elif(x_train[k][i]== 0 and y_train[k]== 1):
#             counttrain0[i][1]+= 1
#         elif(x_train[k][i]== 1 and y_train[k]== 0):
#             counttrain1[i][0]+= 1
#         elif(x_train[k][i]== 1 and y_train[k]== 1):
#             counttrain1[i][1]+= 1
#
#
# print (counttrain0[0], counttrain0[10], counttrain0[20])
# print (counttrain1[:, 0][:100])
# counttrain0.dump('./counttrain0.mat')
# counttrain1.dump('./counttrain1.mat')#fine prolly
counttrain0= np.load('./counttrain0.mat')
counttrain1= np.load('./counttrain1.mat')

a= y_train.count(0)
b= y_train.count(1)
priortrouser= a/(a+b)
priorpullover= b/(a+b)

for i in range(len(counttrain0)):
    for j in range(2):
        if(j== 0):
            counttrain0[i][j]/= a
            counttrain1[i][j]/= a
        else:
            counttrain0[i][j]/= b
            counttrain1[i][j]/= b


print ("The prior trouser is: ", priortrouser)
print ("The prior pullover is: ", priorpullover)

y_pred= np.zeros(shape= (5, len(x_test)))
thresholds= [0.3, 0.4, 0.5, 0.6, 0.7]
for k in range(len(thresholds)):
    for i in range(len(x_test)):
        pdt1= 0
        pdt2= 0
        for j in range(len(x_test[0])):
            if(x_test[i][j]== 1):
                pdt1+= math.log(counttrain1[j][0])#for trousers
                pdt2+= math.log(counttrain1[j][1])#for pullovers
            else:
                pdt1+= math.log(counttrain0[j][0])
                pdt2+= math.log(counttrain0[j][1])

        prob0= (pdt1*priortrouser)/(pdt1*priortrouser + pdt2*priorpullover)#chnge this
        if(prob0<= thresholds[k]):
            y_pred[k][i]=0
        else:
            y_pred[k][i]=1

print ("Lengths: ", len(y_pred), len(y_test))
print (y_pred[:, 0], y_pred[:, 1])
#print (y_pred)
tparray= []
fparray= []
for i in range(5):
    confmatrix = np.zeros(shape=(2, 2))
    tp= 0
    fp= 0
    fn= 0
    tn= 0
    for j in range(len(y_test)):
        if(y_pred[i][j]== 0 and y_test[j]== 0):
            tp+= 1
        elif(y_pred[i][j]== 1 and y_test[j]== 1):
            tn+= 1
        elif(y_pred[i][j]== 0 and y_test[j]== 1):
            fp+= 1
        elif(y_pred[i][j]== 1 and y_test[j]== 0):
            fn+= 1

    confmatrix[0][0]= tp
    confmatrix[1][1]= tn
    confmatrix[0][1]= fp
    confmatrix[1][0]= fn
    tparray.append(tp/(tp+fn))
    fparray.append(fp/(fp+tn))
    print ("CASE" + str(i+1) + " :-")
    print (confmatrix)
    print ("The precision is: " + str((tp/(tp+fp))))
    print ("The recall is: " + str((tp/(tp+fn))))

plt.plot(fparray, tparray)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("ROC Curve")
plt.show()

