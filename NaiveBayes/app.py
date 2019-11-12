# Import packages
import cffi
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sys import argv
from time import time
from classificationNB_Pynq import Naivebayes

# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
dataset = "CARDIO"
decision = 1
_accel_ = 1

train_file = "crd.d"
test_file = "cardtest.d"

# with open(dataset, 'r') as f:
#     for line in f:
#         if line[0] != '#':
#             parameters = line.split(',')
#             numClasses = int(parameters[0])
#             numFeatures = int(parameters[1])
# f.close() 
with open(dataset, 'r') as f:
    for line in f:
        if line[0] != '#':
            parameters = line.split(',')
            numClasses = int(parameters[0])
            numFeatures = int(parameters[1])
f.close()  

print("* NaiveBayes Application *")
print(" # train file:               {:s}".format(train_file))
print(" # test file:                {:s}".format(test_file))

_accel_ = 0 #int(input("Select mode (0: SW-only, 1: HW accelerated): "))


from pyspark.mllib.regression import LabeledPoint

def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """

    data = [float(s) for s in line.split(',')]

    return LabeledPoint(data[0], data[1:])

trainSet = []
with open(train_file, 'r') as f:
    for line in f:
        trainSet.append(parsePoint(line))
f.close()

testSet = []
with open(test_file, 'r') as f:
    for line in f:
        testSet.append(parsePoint(line))
f.close() 

NB = Naivebayes(numClasses, numFeatures, decision)


start_mt = time()
NB.train(trainSet, _accel_)
end_mt = time()
print("! Time running Naive Bayes train in software: {:.3f} sec".format(end_mt - start_mt))

start_mp = time()    
NB.test(testSet,_accel_)    
end_mp = time()
print("! Time running Naive Bayes test in harware: {:.3f} sec".format(end_mp - start_mp))


label = []
train = []
test = []
labtest = []
for i in trainSet:
    label.append(i.label)
    train.append(i.features)
for i in testSet:
    labtest.append(i.label)
    test.append(i.features)

# from pyspark.mllib_accel.classificationNB_Pynq import Naivebayes
# from sys import argv
# from time import time


# NB = Naivebayes(numClasses, numFeatures, decision)

# start = time()

# # Train a Naive Bayes model given an dataset of (label, features) pairs. 
# NB.train(trainSet, _accel_)

# end = time()

# if _accel_:
#     print("! Time running Naive Bayes train in hardware: {:.3f} sec".format(end - start))
# else:
#     print("! Time running Naive Bayes train in software: {:.3f} sec".format(end - start))

# stats = ["Means","Variances","Priors"]
# for i in range (3):
#     NB.save("outputs/trainPack"+ stats[i] + ".txt", i)


# # Import data
# training = pd.read_table('crd.d')
# test = pd.read_csv('cardtest.d')


# # Create the X, Y, Training and Test
# xtrain = training.drop(index=33)
# print(xtrain)
# ytrain = training.loc[1:]
# print(ytrain)
# xtest = test.drop(index=10)
# ytest = test.loc[1:]


# Init the Gaussian Classifier
model = GaussianNB()
start1 = time()
start_t = time()
# train = [col[0] for col in train]
# label = [col[1:] for col in train]
# Train the model 
model.fit(train, label)
end_t = time()

print("! Time running Naive Bayes train-2 in harware: {:.3f} sec".format(end_t - start_t))
# Predict Output 
start_p = time()
pred = model.predict(test)

# print(pred)
end_p = time()
print("! Time running Naive Bayes test-2 in harware: {:.3f} sec".format(end_p - start_p))

end1 = time()
print("! Time running Naive Bayes overall in harware: {:.3f} sec".format(end1 - start1))

mat = confusion_matrix(pred,labtest) 
print(mat)
j = 1
summ = 0
for i in range(len(mat)):
    summ += mat[i,i] 
         
print(sum(sum(mat)))
print(summ/(sum(sum(mat)))*100)

print("speedup_pred = ", (start_mp-end_mp)/(start_p-end_p), "\nspeedup_train = ", (start_mt-end_mt)/(start_t-end_t))
