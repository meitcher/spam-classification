function [vocabList, vocabIndex] = getVocabList(fileName, vocab_size)

	fid = fopen(fileName);
	n = vocab_size; 
	vocabList = [];
	vocabIndex = cell(vocab_size, 1);

	for i = 1:n
		fscanf(fid, '%d', 1);
		t = fscanf(fid, '%s', 1);
		vocabList = setfield(vocabList, t, i);
		vocabIndex(i) = t;
		fprintf('%4d of %d\r', i, n);
	end

	fclose(fid);
end
