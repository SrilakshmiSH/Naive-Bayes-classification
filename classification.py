
# Programming assignment 2

import random
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# to count spam and non-spam data
countspam = countnotspam = 0.0
# values to be calculated from confusion matrix
true_positive = true_negative = false_positive = false_negative = 0.0
# obtain classification value
classifcation = 0

# read Spambase dataset
training_data = pd.read_csv("/Users/srilakshmishivakumar/cs554 ml/prg2/spambase/spambase.data", header=None, dtype=float);
np_data = training_data.as_matrix();

# 1813 rows of entire dataset are spam
spam = np_data[:1813,:]
i1 = np.arange(spam.shape[0])
np.random.shuffle(i1)

# 2788 rows of entire dataset are non-spam
notspam = np_data[1813:,:]
i2 = np.arange(notspam.shape[0])
np.random.shuffle(i2)

# train dataset consists of 40% spam data and 60% non-spam data
trainspam = spam[:906,:]
trainnotspam = notspam[:1394,:]

# test dataset consists of 40% spam data and 60% non-spam data
testspam = spam[906:,:]
testnotspam = notspam[1394:,:]

# concatenate the spam and non-spam into final train dataset
finaltrain = np.concatenate((trainspam,trainnotspam),axis=0)
# labels of train datset
finaltrain_target = finaltrain[:,57]

# concatenate the spam and non-spam into final test dataset
finaltest = np.concatenate((testspam,testnotspam),axis=0)
#labels of test dataset
finaltest_target = finaltest[:,57]

# function to implement Naïve Bayes classifcation to classify the test dataset
def formula(mean,std,a):
	np.seterr(divide='ignore')
	part1 = float(1 / (std * (np.sqrt(2 * np.pi))))
	part2 = float(np.exp(-1 * (np.square(a - mean))/(2 * np.square(float(std * std)))))
	res = part1 * part2
	return res

# calculate the prior probabilities of spam and non-spam data
for i in range(0,finaltrain.shape[0]):
	if(finaltrain[i,57] == 1):
		countspam += 1
	else:
		countnotspam += 1

priorspam = countspam / len(finaltrain);
#print(priorspam)
priornotspam = countnotspam / len(finaltrain);
#print(priornotspam)

# list to calculate mean and standard deviation for spam data
spam_mean = []
spam_sd = []

# list to calculate mean and standard deviation for non-spam data
notspam_mean = []
notspam_sd = []

# append spam features to spam array and non-spam features to notspam array
# calculate mean and standard deviation of spam and non-spam data for every feature
# Change standard deviation to 0.0001 if its 0 to avoid divide by zero error

for i in range(0,finaltrain.shape[1]):
	spamarray = []
	notspamarray = []

	for j in range(finaltrain.shape[0]):
		if (finaltrain[j][-1] == 1):
			spamarray.append(finaltrain[j][i])
		else:
			notspamarray.append(finaltrain[j][i])

	spam_mean.append(np.mean(spamarray))
	spam_sd.append(np.std(spamarray))
	notspam_mean.append(np.mean(notspamarray))
	notspam_sd.append(np.std(notspamarray))

for k in range(len(spam_sd)):
	if (spam_sd[k] == 0):
		spam_sd[k] = 0.0001

	if (notspam_sd[k] == 0):
		notspam_sd[k] = 0.0001

# to obtain the classification result after Gaussian Naïve Bayes calculation
classification_result = []

# classify the test datatset using Gaussian Naïve Bayes formula
# log is used since the values are very small
for i in range(finaltest.shape[0]):
	temp1 = np.log(priorspam)
	temp2 = np.log(priornotspam)

# to obtain the classification value
	for j in range(0,57):
		a = finaltest[i][j]	
		temp1 += np.log(formula(spam_mean[j], spam_sd[j], a))
		temp2 += np.log(formula(notspam_mean[j], notspam_sd[j], a))
	classification = np.argmax([temp2, temp1])
	classification_result.append(classification)

# to obtain confusion matrix using target values of test dataset and the classification result
confusion_matrix = confusion_matrix(finaltest_target, classification_result)
print("\nConfusion matrix\n",confusion_matrix)

for i in range(len(classification_result)):
	if (classification_result[i] == 1 and finaltest_target[i] == 1):
		true_positive += 1
	elif (classification_result[i] == 0 and finaltest_target[i] == 0):
		true_negative += 1
	elif (classification_result[i] == 1 and finaltest_target[i] == 0):
		false_positive += 1
	else:
		false_negative += 1

# to calculate accuracy, precision and recall
accuracy = float(true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive)
precision = float(true_positive) / (true_positive + false_positive)
recall = float(true_positive) / (true_positive + false_negative)

print("\nAccuracy - ",accuracy)
print("Precision - ",precision)
print("Recall - ",recall)
