import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import math
from mlxtend.data import loadlocal_mnist


X_train, Y_train= loadlocal_mnist(images_path='./train-images.idx3-ubyte', labels_path='./train-labels.idx1-ubyte')
X_test, Y_test= loadlocal_mnist(images_path='./t10k-images.idx3-ubyte', labels_path='./t10k-labels.idx1-ubyte')#fine
x_train= []
y_train= []
x_test= []
y_test= []

for i in range(len(X_train)):
    if(Y_train[i]== 1 or Y_train[i]== 8):
        x_train.append(X_train[i])
        y_train.append(Y_train[i])

for i in range(len(X_test)):
    if(Y_test[i]== 1 or Y_test[i]== 8):
        x_test.append(X_test[i])
        y_test.append(Y_test[i])

print (len(x_train), len(y_train), len(x_test), len(y_test))
for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        if(x_test[i][j]<= 127):
            x_test[i][j]= 0
        else:
            x_test[i][j]= 1

for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        if(x_train[i][j]<= 127):
            x_train[i][j]= 0
        else:
            x_train[i][j]= 1


#RANDOM 5-FOLD ---------------------------------------------------------------------------------------------------
# a= np.arange(len(x_train))
# np.random.shuffle(a)
# a.dump('./shuffled.mat')
a= np.load('./shuffled.mat')
b= len(x_train)//5
validationaccuracy= []
for i in range(5):
    validation_train= []
    validation_test= []
    xx_train= []
    yy_train= []
    for j in range(i*b, (i+1)*b):
        validation_train.append(x_train[a[j]])
        validation_test.append(y_train[a[j]])

    for j in range(len(x_train)):
        if(j< i*b or j>= (i+1)*b):
            xx_train.append(x_train[a[j]])
            yy_train.append(y_train[a[j]])

    print (len(validation_train), len(validation_test), len(xx_train), len(yy_train))
    #count matrices build!
    counttrain0= np.ones(shape= (784, 2))
    counttrain1= np.ones(shape= (784, 2))
    counttrain0= counttrain0*(1.0/100.0)
    counttrain1= counttrain1*(1.0/100.0)
    for j in range(len(counttrain0)):
        for k in range(len(xx_train)):
            if(xx_train[k][j]== 0 and yy_train[k]== 1):
                counttrain0[j][0]+= 1
            elif(xx_train[k][j]== 0 and yy_train[k]== 8):
                counttrain0[j][1]+= 1
            elif(xx_train[k][j]== 1 and yy_train[k]== 1):
                counttrain1[j][0]+= 1
            elif(xx_train[k][j]== 1 and yy_train[k]== 8):
                counttrain1[j][1]+= 1

    c = yy_train.count(1)
    d = yy_train.count(8)
    prior1= c/(c+d)
    prior8= b/(c+d)

    for j in range(len(counttrain0)):
        for k in range(2):
            if (k == 0):
                counttrain0[j][k]/= c
                counttrain1[j][k]/= c
            else:
                counttrain0[j][k]/= d
                counttrain1[j][k]/= d

    yy_pred= []
    for j in range(len(validation_train)):
        pdt1= 0
        pdt8= 0
        for k in range(len(validation_train[0])):
            if(validation_train[j][k]== 0):
                pdt1+= math.log(counttrain0[k][0])
                pdt8+= math.log(counttrain0[k][1])
            else:
                pdt1+= math.log(counttrain1[k][0])
                pdt8+= math.log(counttrain1[k][1])

        if(pdt1*prior1 >= pdt8*prior8):
            yy_pred.append(1)
        else:
            yy_pred.append(8)

    count= 0
    for j in range(len(validation_train)):
        if(yy_pred[j]== validation_test[j]):
            count+= 1

    validationaccuracy.append(count/len(validation_train))
    print(count/len(validation_train))

print (validationaccuracy)
avg= np.sum(validationaccuracy)/len(validationaccuracy)
print ("The average accuracy is:- " + str(avg))
deviation= 0
for i in range(len(validationaccuracy)):
    deviation+= (validationaccuracy[i]-avg)**2

print ("The standard deviation is:- " + str(deviation/len(validationaccuracy)))
print ("Clearly the accuracy is the highest for the second case.")


#---------------DOING THE SAME FOR THE BEST CASE FOR THE ROC AND DET CRUVES-----------------


validation_train= []
validation_test= []
xx_train= []
yy_train= []
for j in range(b, 2*b):
    validation_train.append(x_train[a[j]])
    validation_test.append(y_train[a[j]])

for j in range(len(x_train)):
    if(j< b or j>= 2*b):
        xx_train.append(x_train[a[j]])
        yy_train.append(y_train[a[j]])

counttrain0= np.ones(shape= (784, 2))
counttrain1= np.ones(shape= (784, 2))
counttrain0= counttrain0*(1.0/100.0)
counttrain1= counttrain1*(1.0/100.0)
for j in range(len(counttrain0)):
    for k in range(len(xx_train)):
        if(xx_train[k][j]== 0 and yy_train[k]== 1):
            counttrain0[j][0]+= 1
        elif(xx_train[k][j]== 0 and yy_train[k]== 8):
            counttrain0[j][1]+= 1
        elif(xx_train[k][j]== 1 and yy_train[k]== 1):
            counttrain1[j][0]+= 1
        elif(xx_train[k][j]== 1 and yy_train[k]== 8):
            counttrain1[j][1]+= 1

c= yy_train.count(1)
d= yy_train.count(8)

for j in range(len(counttrain0)):
    for k in range(2):
        if (k == 0):
            counttrain0[j][k]/= c
            counttrain1[j][k]/= c
        else:
            counttrain0[j][k]/= d
            counttrain1[j][k]/= d

prior1= c/(c+d)
prior8= b/(c+d)

y_pred= np.zeros(shape= (5, len(validation_train)))
thresholds= [0.3, 0.4, 0.5, 0.6, 0.7]
for i in range(len(thresholds)):
    for j in range(len(validation_train)):
        pdt1= 0
        pdt8= 0
        for k in range(len(validation_train[0])):
            if(validation_train[j][k]== 0):
                pdt1+= math.log(counttrain0[k][0])
                pdt8+= math.log(counttrain0[k][1])
            else:
                pdt1+= math.log(counttrain1[k][0])
                pdt8+= math.log(counttrain1[k][1])
        prob1= pdt1*prior1/(pdt1*prior1 + pdt8*prior8)
        if(prob1<= thresholds[i]):
            y_pred[i][j]= 1
        else:
            y_pred[i][j]= 8

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
        if(y_pred[i][j]== 1 and validation_test[j]== 1):
            tp+= 1
        elif(y_pred[i][j]== 8 and validation_test[j]== 8):
            tn+= 1
        elif(y_pred[i][j]== 1 and validation_test[j]== 8):
            fp+= 1
        elif(y_pred[i][j]== 8 and validation_test[j]== 1):
            fn+= 1

    confmatrix[0][0] = tp
    confmatrix[1][1] = tn
    confmatrix[0][1] = fp
    confmatrix[1][0] = fn
    tparray.append(tp / (tp + fn))
    fparray.append(fp / (fp + tn))
    fnarray.append(fn/(fn+tp))
    print(confmatrix)

#calculating the equal error rate
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



