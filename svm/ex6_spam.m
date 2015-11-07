clear ; close all; clc

n = 45525;
[vocabList, vocabIndex] = getVocabList('vocabulary/enron1_vocabulary.txt', n);
% [X, y] = getEnron1(vocabList, n);
% save enron1.mat X y
load('enron1.mat')

[m, asd] = size(X)
t = floor(m - 0.8*m)
v = floor(t - 0.5*t)

Xtrain = X(1:t);
Xval = X(t:t+v);
Xtest = X(t+v:m);
ytrain = y(1:t);
yval = y(t:t+v);
ytest = y(t+v:m);


fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(Xtrain, ytrain, C, @linearKernel);

p = svmPredict(model, Xtrain);
fprintf('Training Accuracy: %f\n', mean(double(p == ytrain)) * 100);

p = svmPredict(model, Xval);
fprintf('Validating Accuracy: %f\n', mean(double(p == yval)) * 100);

p = svmPredict(model, Xtest);
fprintf('Testing Accuracy: %f\n', mean(double(p == ytest)) * 100);



% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabIndex(idx(i)), weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;



filename = 'emailSample3.txt';
file_contents = readFile(filename);
word_indices  = processEmail(file_contents, vocabList);
x             = emailFeatures(word_indices, n);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

