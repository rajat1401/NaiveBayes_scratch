# NaiveBayes_scratch
Implemented Naive Bayes from scratch with Cross Validation (K-fold and stratified Kfold) on MNIST dataset.

	ASSIGNMENT 1 REPORT

Ques1:
The Naive Bayes classifier is built using counts of 1’s and 0’s for each pixel values(considered as features for the images post Binarization) and counts for each value(0 or 1) are kept in matrices with # of columns same as the # of classes for prediction.  These counts are then normalized to not make the predictions biased. 
Bayes Rule: 
	P(class|sequence)  ∝ P(sequence|class)*P(class)
	P(sequence|class) = 𝚷 P(sequence[i]|class)
NOTE:
Since some counts can be zero, add-one smoothing is used to obtain non-zero probabilities.
Now, due to the add-one smoothing, since the final probabilities might not sum up to one, we normalize them one more time to obtain the final probabilities. 

THE ABOVE GENERAL PROCEDURE IS ALSO FOLLOWED FOR PART B WHERE 10 CLASSES ARE ENCOUNTERED.
                        
          

Once we have the predictions, we can calculate the # of True Positives, True Negatives, False Positives, and False Negatives. Once built the confusion matrix, we can calculate the TP rate and FP rate and plot the curve using the plt.plot function in python. 
Precision= TP/(TP+FP)
Recall= TP/(TP+FN)
	





Ques2:
K-Fold is performed by randomly shuffling the training dataset using np.random.shuffle function and folds are created of the same size. Validation dataset is chosen to be of one of the folds in a dataset one by one, and training is done on the rest of the 4 folds. The similar procedure as above is then performed on the 4 folds and tested on the validation data giving validation accuracies(5). The average and standard deviations are then computed as usual. The best accuracy is obtained when validation is done on the second fold (67%) and the ROC and DET curves are plotted for this model along with the confusion matrix

 


For stratified K-Fold, one simple addition is that the folds are randomly created but also have the same proportions of both the classes as the original training dataset. Then the similar procedure as above is followed. All the validation accuracies are same to the decimal, so the best model can be trained on any fold. The validation accuracies come out to be about 68.5% and the ROC and DET curves are plotted for the model along with the confusion matrix.

  

      


2). The training and validation accuracies for classes 3 and 8 come out to be poorer than 1 and 8 since the ratio of training samples between 3 and 8 is closer to 1 than the ratio for classes 1 and 8 which is why it becomes really tough for the model to make decisions since the final probabilities are very cut to cut (we get a decision boundary with small margin.)



