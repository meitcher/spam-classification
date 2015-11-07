function [X, y] = getEnron1(vocabList, n)

	fileList_ham = getAllFiles('dados/enron1/ham/');
	fileList_spam = getAllFiles('dados/enron1/spam/');

	a = length(fileList_ham);
	b = length(fileList_spam);
	m = a + b; 


	fprintf('\n%d %d\n', m, n)
	X = zeros(m, n);
	y = zeros(m, 1);
	y(b:m) = 1;

	for i=1:m
		if i > a
			k = sprintf('dados/enron1/spam/%s', fileList_spam(i-a, :));
		else
			k = sprintf('dados/enron1/ham/%s', fileList_ham(i, :));
		end
		file_contents = readFile(k);
		word_indices  = processEmail(file_contents, vocabList);
		features      = emailFeatures(word_indices, n);
		X(i, :) 	  = features';
		fprintf('%4d of %d\r', i, m)
	end


end
