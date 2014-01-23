import numpy as np
from sklearn import svm
from sklearn import grid_search
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(data, groundtruthLabels, predictedLabels):
	"""
	Visualize first two dims of the data

	Color stands for the predicted label
	Shape stands for the groundtruth
	"""

	indexTP = [i for i in range(len(groundtruthLabels)) \
					if (predictedLabels[i]==1 and groundtruthLabels[i]==1)]
	indexTN = [i for i in range(len(groundtruthLabels))\
					if (predictedLabels[i]==0 and groundtruthLabels[i]==0)]
	indexFP = [i for i in range(len(groundtruthLabels))\
					if (predictedLabels[i]==1 and groundtruthLabels[i]==0)]
	indexFN = [i for i in range(len(groundtruthLabels))\
					if (predictedLabels[i]==0 and groundtruthLabels[i]==1)]
	cols = [i in [0,1] for i in range(data.shape[1])]

	dataTP = data[indexTP, 0:2]
	dataTN = data[indexTN, 0:2]
	dataFP = data[indexFP, 0:2]
	dataFN = data[indexFN, 0:2]

	plt.scatter(dataTP[:,0].ravel(), dataTP[:,1].ravel(), c='b', marker='o', label="TP, 11")
	plt.scatter(dataTN[:,0].ravel(), dataTN[:,1].ravel(), c='b', marker='1', label="TN, 00")
	plt.scatter(dataFP[:,0].ravel(), dataFP[:,1].ravel(), c='r', marker='o', label="FP, 01")
	plt.scatter(dataFN[:,0].ravel(), dataFN[:,1].ravel(), c='r', marker='1', label="FN, 10")
	# axis
	#plt.set_xlabel('Dim 1')
	#plt.set_ylabel('Dim 2')
	#plt.set_zlabel('Dim 3')
	# legend
	plt.legend()

	plt.show()
	

def crossValidation(train, trainLabels, fold, clf):
	meanPerformance = 0;
	nFold = (int)(len(trainLabels) / fold)
	for i in range(fold):
		trainPicked = np.concatenate(\
				(train[0:i*nFold,],\
				 train[(i+1)*nFold:len(train),]))
		trainLabelsPicked = np.concatenate(\
				(trainLabels[0:i*nFold],\
				 trainLabels[(i+1)*nFold:len(trainLabels),]))
		testPicked = train[i*nFold:(i+1)*nFold,]
		testLabelsPicked = trainLabels[i*nFold:(i+1)*nFold]

		predictions = runRoutine(trainPicked, trainLabelsPicked, testPicked, clf)
		err = evaluate(predictions, testLabelsPicked)
		meanPerformance += err / fold

		print(err)
		visualize(testPicked, testLabelsPicked, predictions)
	
	print("The mean error is: ", meanPerformance)
	return meanPerformance

def searchParams(train, trainLabels):
	C_range = 10.0 ** np.arange(6.5, 6.75, 0.25)
	gamma_range = 10.0 ** np.arange(-1.5, -1.25, 0.25)
	params = dict(gamma=gamma_range,C=C_range)
	cvk = StratifiedKFold(trainLabels, n_folds=3)
	classifier = svm.SVC()
	clf = grid_search.GridSearchCV(classifier,param_grid=params,cv=5)
	clf.fit(train, trainLabels)
	print(clf.best_estimator_)
	return clf.best_estimator_

def runRoutine(train, trainLabels, test, clf):
	clf.fit(train, trainLabels)

	predictions = clf.predict(test)

	return predictions

def evaluate(predictions, labels):
	rst = sum(np.abs(predictions-labels)) / len(predictions)

	return rst

def trainFeatureSelectionModel(data):
	model = PCA(n_components=12,whiten=True)
	model.fit(data)
	return model

def featureSelection(data, model):
	return model.transform(data)

def loadData():
	train = np.loadtxt(open("data/train.csv", "rb"), delimiter=",", skiprows=0)
	trainLabels = np.loadtxt(open("data/trainLabels.csv", "rb"), delimiter=",", skiprows=0)
	test = np.loadtxt(open("data/test.csv", "rb"), delimiter=",", skiprows=0)
	return (train, trainLabels, test)

def work(clf):
	(train, trainLabels, test) = preprocess()
	predictions = runRoutine(train, trainLabels, test, clf)
	f = open('submission/linearSVMSubmittion.csv', 'wb')
	f.write('Id,Solution\n')
	for i in range(len(predictions)):
		f.write('%d,%d\n' % (i+1, predictions[i]))
	
def test(clf):
	(train, trainLabels, test) = preprocess()
	#(train, trainLabels, test) = loadData()
	crossValidation(train, trainLabels, 10, clf)

def preprocess():
	(train, trainLabels, test) = loadData()
	# Use all data as a whole
	#featSelModel = trainFeatureSelectionModel(np.concatenate((train,test)))
	# Use training data only
	featSelModel = trainFeatureSelectionModel(train)

	train = featureSelection(train, featSelModel)
	test = featureSelection(test, featSelModel)

	return (train, trainLabels, test)


# for execfile
(train, trainLabels, test) = preprocess()