%[X, Y, test] = loadData();

class = knnclassify(test,X,Y,3);

genSubmission(class, 'knn-naive');