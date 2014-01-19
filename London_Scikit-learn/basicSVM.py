import numpy as np
from sklearn import svm
from sklearn import grid_search
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
import matplotlib as plt

def crossValidation(train, trainLabels, fold):
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

		predictions = runRoutine(trainPicked, trainLabelsPicked, testPicked)
		err = evaluate(predictions, testLabelsPicked)
		print(err)
		meanPerformance += err / fold
	
	print("The mean error is: ", meanPerformance)
	return meanPerformance

def searchParams(train, trainLabels):
	C_range = 10.0 ** np.arange(7,10)
	gamma_range = 10.0 ** np.arange(-4,0)
	params = dict(gamma=gamma_range,C=C_range)
	cvk = StratifiedKFold(trainLabels, n_folds=6)
	classifier = svm.SVC()
	clf = grid_search.GridSearchCV(classifier,param_grid=params,cv=5)
	clf.fit(train, trainLabels)
	print(clf.best_estimator_)

def runRoutine(train, trainLabels, test):
	clf = svm.SVC(gamma=0.0001, C=316227)
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

def work():
	(train, trainLabels, test) = preprocess()
	predictions = runRoutine(train, trainLabels, test)
	f = open('submission/linearSVMSubmittion.csv', 'wb')
	f.write('Id,Solution\n')
	for i in range(len(predictions)):
		f.write('%d,%d\n' % (i+1, predictions[i]))
	
def test():
	(train, trainLabels, test) = preprocess()
	#(train, trainLabels, test) = loadData()
	crossValidation(train, trainLabels, 10)

def preprocess():
	(train, trainLabels, test) = loadData()
	# Use all data as a whole
	#featSelModel = trainFeatureSelectionModel(np.concatenate((train,test)))
	# Use training data only
	featSelModel = trainFeatureSelectionModel(train)

	train = featureSelection(train, featSelModel)
	test = featureSelection(test, featSelModel)

	return (train, trainLabels, test)