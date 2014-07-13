function [X, Y, test] = loadData()

X = csvread('../data/train.csv', 1, 0);
Y = X(:,1);
X(:,1) = [];

test = csvread('../data/test.csv', 1, 0);