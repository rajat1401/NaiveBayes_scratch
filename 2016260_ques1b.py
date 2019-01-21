import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle
from mlxtend.data import loadlocal_mnist


x_train, y_train= loadlocal_mnist(images_path='./train-images.idx3-ubyte', labels_path='./train-labels.idx1-ubyte')
x_test, y_test= loadlocal_mnist(images_path='./t10k-images.idx3-ubyte', labels_path='./t10k-labels.idx1-ubyte')#fine

# LIMIT= 127
# for i in range(len(x_train)):
#     for j in range(len(x_train[0])):
#         if(x_train[i][j]<= LIMIT):
#             x_train[i][j]= 0
#         else:
#             x_train[i][j]= 1
#
# for i in range(len(x_test)):
#     for j in range(len(x_test[0])):
#         if(x_test[i][j]<= LIMIT):
#             x_test[i][j]= 0
#         else:
#             x_test[i][j]= 1


# with open('./dectrain.pkl', 'wb+') as f:
#     pickle.dump(x_train, f)
# with open('./dectest.pkl', 'wb+') as f:
#     pickle.dump(x_test, f)
f= open('./dectrain.pkl', 'rb')
x_train= pickle.load(f)
f= open('./dectest.pkl', 'rb')
x_test= pickle.load(f)


# counttrain0= np.ones(shape= (784, 10))
# counttrain1= np.ones(shape= (784, 10))
# for i in range(len(counttrain0)):
#     for j in range(len(x_train)):
#         if(x_train[j][i]== 0):
#             counttrain0[i][y_train[j]]+= 1
#         else:
#             counttrain1[i][y_train[j]]+= 1
#
# print (counttrain0[0], counttrain0[10])
# print (counttrain1[0], counttrain1[10])
# counttrain0.dump('./counttrainnext0.mat')
# counttrain1.dump('./counttrainnext1.mat')
counttrain0= np.load('./counttrainnext0.mat')
counttrain1= np.load('./counttrainnext1.mat')

counts= []
for i in range(10):
    counts.append(list(y_train).count(i))

priors= []
a= sum(counts)
for i in range(10):
    priors.append(counts[i]/a)

for i in range(len(counttrain0)):
    for j in range(10):
        counttrain0[i][j]/= counts[j]
        counttrain1[i][j]/= counts[j]

thresholds= [0.08, 0.09, 0.10, 0.11, 0.12]
for i in range(10):
    print ("CLASS" + str(i) + ".............................................................................")
    y_pred= np.zeros(shape= (5, len(y_test)))
    for j in range(len(thresholds)):
        for k in range(len(x_test)):
            pdts= [0.0]*10
            for l in range(len(x_test[0])):
                if(x_test[k][l]== 1):
                    pdts[0]+= math.log(counttrain1[l][0])
                    pdts[1]+= math.log(counttrain1[l][1])
                    pdts[2]+= math.log(counttrain1[l][2])
                    pdts[3]+= math.log(counttrain1[l][3])
                    pdts[4]+= math.log(counttrain1[l][4])
                    pdts[5]+= math.log(counttrain1[l][5])
                    pdts[6]+= math.log(counttrain1[l][6])
                    pdts[7]+= math.log(counttrain1[l][7])
                    pdts[8]+= math.log(counttrain1[l][8])
                    pdts[9]+= math.log(counttrain1[l][9])
                else:
                    pdts[0]+= math.log(counttrain0[l][0])
                    pdts[1]+= math.log(counttrain0[l][1])
                    pdts[2]+= math.log(counttrain0[l][2])
                    pdts[3]+= math.log(counttrain0[l][3])
                    pdts[4]+= math.log(counttrain0[l][4])
                    pdts[5]+= math.log(counttrain0[l][5])
                    pdts[6]+= math.log(counttrain0[l][6])
                    pdts[7]+= math.log(counttrain0[l][7])
                    pdts[8]+= math.log(counttrain0[l][8])
                    pdts[9]+= math.log(counttrain0[l][9])

            for l in range(10):
                pdts[l]*= priors[l]#calculating pobabilities

            b= sum(pdts)
            for l in range(10):
                pdts[l]/= b#normalize

            if(pdts[i]<= thresholds[j]):
                y_pred[j][k]= i
            else:
                y_pred[j][k]= -1#something to denote that the prediction not equal to the ith class.

    #print(y_pred[:, 0], y_pred[:, 1])
    tparray= []
    fparray= []
    for j in range(5):
        confmatrix= np.zeros(shape= (2,2))
        tp= 0
        fp= 0
        fn= 0
        tn= 0
        for k in range(len(y_test)):
            if(y_pred[j][k]== i and y_test[k]== i):
                tp+= 1
            elif(y_pred[j][k]== -1 and y_test[k]!= i):
                tn+= 1
            elif(y_pred[j][k]== i and y_test[k]!= i):
                fp+= 1
            elif(y_pred[j][k]== -1 and y_test[k]== i):
                fn+= 1

        confmatrix[0][0] = tp
        confmatrix[1][1] = tn
        confmatrix[0][1] = fp
        confmatrix[1][0] = fn
        tparray.append(tp/(tp+fn))
        fparray.append(fp/(fp+tn))
        print ('CASE' + str(j+1) + " :-")
        print (confmatrix)
        print ("The precision is :" + str(tp/(tp+fp)))
        print ("The recall is :" + str(tp/(tp+fn)))

    plt.plot(fparray, tparray)


plt.plot([0,1], [0,1], linestyle= 'dashed')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.show()





