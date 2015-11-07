function x = emailFeatures(word_indices, vocabLen)
	x = zeros(vocabLen, 1);
	x(word_indices) = 1;
end
