
import csv
with open('iris.data', 'rb') as csvfile:
	lines = csv.reader(csvfile)
	for row in lines:
		print ', '.join(row)


import csv
import random
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for X in range(len(dataset)-1):
	        for y in range(4):
	            dataset[X][y] = float(dataset[X][y])
	        if random.random() < split:
	            trainingSet.append(dataset[X])
	        else:
	            testSet.append(dataset[X])

import math
def euclideanDistance(instance1,instance2,length):
	dist=0
	for X in range(length):
		dist +=pow((instance1[X]-instance2[X]),2)
	return math.sqrt(dist)

d1 = [2,2,2,'a']
d2=[4,4,4,'b']
dist =euclideanDistance(d1,d2,3)
print ('distance:'+ repr(dist))


import operator
def get_Neighbors(trainingSet,testInstance,k):
	dists=[]
	length=len(testInstance)-1
	for X in range(len(trainingSet)):
		dist=euclideanDistance(testInstance,trainingSet[X],length)
		dists.append((trainingSet[X],dist))
	dists.sort(key=operator.itemgetter(1))
	neighbors=[]
	for x in range(k):
		neighbors.append(dists[x][0])
	return neighbors



#Test the getNeighbors function 
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = get_Neighbors(trainSet, testInstance, 1)
print(neighbors)

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
#test the getResponse function 


neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
response = getResponse(neighbors)
print(response)

#calculate accuracy of predictions 

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)


def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	k = 3
	for X in range(len(testSet)):
		neighbors = get_Neighbors(trainingSet, testSet[X], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[X][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')


main()