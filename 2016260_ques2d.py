#ACTUALLY ITS 2B FR STRATIFIED, BUT DOESNT MATTER

import numpy as np
import matplotlib .pyplot as plt
import pandas as pd
import math
import pickle
from mlxtend.data import loadlocal_mnist


X_train, Y_train= loadlocal_mnist(images_path='./train-images.idx3-ubyte', labels_path='./train-labels.idx1-ubyte')
X_test, Y_test= loadlocal_mnist(images_path='./t10k-images.idx3-ubyte', labels_path='./t10k-labels.idx1-ubyte')#fine
x_train1= []
y_train1= []
x_train8= []
y_train8= []
x_test= []
y_test= []

for i in range(len(X_train)):
    if(Y_train[i]== 3):
        x_train1.append(X_train[i])
        y_train1.append(Y_train[i])
    elif(Y_train[i]== 8):
        x_train8.append(X_train[i])
        y_train8.append(Y_train[i])

for i in range(len(X_test)):
    if(Y_test[i]== 3 or Y_test[i]== 8):
        x_test.append(X_test[i])
        y_test.append(Y_test[i])

for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        if(x_test[i][j]<= 127):
            x_test[i][j]= 0
        else:
            x_test[i][j]= 1

for i in range(len(x_train1)):
    for j in range(len(x_train1[0])):
        if(x_train1[i][j]<= 127):
            x_train1[i][j]= 0
        else:
            x_train1[i][j]= 1

for i in range(len(x_train8)):
    for j in range(len(x_train8[0])):
        if(x_train8[i][j]<= 127):
            x_train8[i][j]= 0
        else:
            x_train8[i][j]= 1

a= len(x_train1) + len(x_train8)
b= a//5#size of each fold
x_train= []
y_train= []
c= len(x_train1)/(len(x_train1) + len(x_train8))
# d= np.arange(len(x_train1))
# np.random.shuffle(d)
# d.dump('./d2.mat')
d= np.load('./d2.mat')
# e= np.arange(len(x_train8))
# np.random.shuffle(e)
# e.dump('./e2.mat')
e= np.load('./e2.mat')
for i in range(5):
    for j in range(int(c*b)):
        x_train.append(x_train1[d[j]])
        y_train.append(y_train1[d[j]])
    for j in range(b-int(c*b)):
        x_train.append(x_train8[e[j]])
        y_train.append(y_train8[e[j]])

validationaccuracy= []
thresholds= [0.3, 0.4, 0.5, 0.6, 0.7]
for i in range(5):
    validation_train= []
    validation_test= []
    xx_train= []
    yy_train= []
    for j in range(i*b, (i+1)*b):
        validation_train.append(x_train[j])
        validation_test.append(y_train[j])

    for j in range(len(x_train)):
        if(j< i*b or j>= (i+1)*b):
            xx_train.append(x_train[j])
            yy_train.append(y_train[j])

    print(len(validation_train), len(validation_test), len(xx_train), len(yy_train))
    counttrain0 = np.ones(shape=(784, 2))
    counttrain1 = np.ones(shape=(784, 2))
    counttrain0 = counttrain0 * (1.0 / 100.0)
    counttrain1 = counttrain1 * (1.0 / 100.0)
    for j in range(len(counttrain0)):
        for k in range(len(xx_train)):
            if (xx_train[k][j] == 0 and yy_train[k] == 3):
                counttrain0[j][0] += 1
            elif (xx_train[k][j] == 0 and yy_train[k] == 8):
                counttrain0[j][1] += 1
            elif (xx_train[k][j] == 1 and yy_train[k] == 3):
                counttrain1[j][0] += 1
            elif (xx_train[k][j] == 1 and yy_train[k] == 8):
                counttrain1[j][1] += 1

    c = yy_train.count(3)
    d = yy_train.count(8)
    prior1 = c / (c + d)
    prior8 = b / (c + d)

    for j in range(len(counttrain0)):
        for k in range(2):
            if (k == 0):
                counttrain0[j][k]/= c
                counttrain1[j][k]/= c
            else:
                counttrain0[j][k]/= d
                counttrain1[j][k]/= d

    yy_pred = []
    for j in range(len(validation_train)):
        pdt1 = 0
        pdt8 = 0
        for k in range(len(validation_train[0])):
            if (validation_train[j][k] == 0):
                pdt1 += math.log(counttrain0[k][0])
                pdt8 += math.log(counttrain0[k][1])
            else:
                pdt1 += math.log(counttrain1[k][0])
                pdt8 += math.log(counttrain1[k][1])

        pdt1+= math.log(prior1)
        pdt8+= math.log(prior8)
        if (pdt1>= pdt8):
            yy_pred.append(1)
        else:
            yy_pred.append(8)

    count = 0
    for j in range(len(validation_train)):
        if (yy_pred[j] == validation_test[j]):
            count += 1


    validationaccuracy.append(count / len(validation_train))
    print(count / len(validation_train))


print (validationaccuracy)
avg= np.sum(validationaccuracy)/len(validationaccuracy)
print ("The average accuracy is:- " + str(avg))
deviation= 0
for i in range(len(validationaccuracy)):
    deviation+= (validationaccuracy[i]-avg)**2

print ("The standard deviation is:- " + str(deviation/len(validationaccuracy)))
print ("The accuracy it seems it same for each case which should be the case.")


#----------------DOING THE SAME FOR THE BEST FOLD------------------------------------

validation_train= x_train[:len(x_train)//5]
validation_test= y_train[:len(y_train)//5]
xx_train= x_train[len(x_train)//5:]
yy_train= y_train[len(y_train)//5:]

counttrain0= np.ones(shape= (784, 2))
counttrain1= np.ones(shape= (784, 2))
counttrain0= counttrain0*(1.0/100.0)
counttrain1= counttrain1*(1.0/100.0)
for i in range(len(counttrain0)):
    for k in range(len(xx_train)):
        if(xx_train[k][i]== 0 and y_train[k]== 3):
            counttrain0[i][0]+= 1
        elif(xx_train[k][i]== 0 and y_train[k]== 8):
            counttrain0[i][1]+= 1
        elif(xx_train[k][i]== 1 and y_train[k]== 3):
            counttrain1[i][0]+= 1
        elif(xx_train[k][i]== 1 and y_train[k]== 8):
            counttrain1[i][1]+= 1

g= y_train.count(3)
h= y_train.count(8)
prior1= g/(g+h)
prior8= h/(g+h)

for i in range(len(counttrain0)):
    for j in range(2):
        if(j== 0):
            counttrain0[i][j]/= g
            counttrain1[i][j]/= g
        else:
            counttrain0[i][j]/= h
            counttrain1[i][j]/= h

yy_pred= np.zeros(shape= (5, len(validation_test)))
for i in range(len(thresholds)):
    for j in range(len(validation_train)):
        pdt1= 0
        pdt8= 0
        for k in range(len(validation_train[0])):
            if(validation_train[j][k]== 1):
                pdt1 += math.log(counttrain1[k][0])  # for trousers
                pdt8 += math.log(counttrain1[k][1])  # for pullovers
            else:
                pdt1 += math.log(counttrain0[k][0])  # for trousers
                pdt8 += math.log(counttrain0[k][1])

        pdt1+= math.log(prior1)
        pdt8+= math.log(prior8)
        prob1= pdt1/(pdt1 + pdt8)
        if(prob1<= thresholds[i]):
            yy_pred[i][j]= 3
        else:
            yy_pred[i][j]= 8

tparray= []
fparray= []
fnarray= []
for i in range(5):
    confmatrix = np.zeros(shape=(2, 2))
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    print ("CASE" + str(i+1) + " :-")
    for j in range(len(y_test)):
        if(yy_pred[i][j]== 3 and validation_test[j]== 3):
            tp+= 1
        elif(yy_pred[i][j]== 8 and validation_test[j]== 8):
            tn+= 1
        elif(yy_pred[i][j]== 3 and validation_test[j]== 8):
            fp+= 1
        elif(yy_pred[i][j]== 8 and validation_test[j]== 3):
            fn+= 1

    confmatrix[0][0] = tp
    confmatrix[1][1] = tn
    confmatrix[0][1] = fp
    confmatrix[1][0] = fn
    tparray.append(tp / (tp + fn))
    fparray.append(fp / (fp + tn))
    fnarray.append(fn / (fn + tp))
    print (confmatrix)

#roc and det curves
min= -100
index= -1
for i in range(len(tparray)):
    if(abs(fnarray[i] - fparray[i])< min):
        min= abs(fparray[i] - fnarray[i])
        index= i

print ("For the threshold of " + str(thresholds[i]) + " the error rates are approx equal!")


plt.plot(fparray, tparray)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("ROC Curve")
plt.show()

plt.plot(fnarray, fparray)
plt.ylabel("False Positive Rate")
plt.xlabel("False Negative Rate")
plt.title("DET Curve")
plt.show()














