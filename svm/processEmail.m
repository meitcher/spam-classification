function word_indices = processEmail(email_contents, vocabList)

word_indices = [];

email_contents = lower(email_contents);
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
email_contents = regexprep(email_contents, '[0-9]+', 'NUMBER');
email_contents = regexprep(email_contents, '(http|https)://[^\s]*', 'HTTPADDR');
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'EMAILADDR');
email_contents = regexprep(email_contents, '[$]+', 'DOLLAR');


while ~isempty(email_contents)

    
    [str, email_contents] = strtok(email_contents, [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
    
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    % try str = porterStemmer(strtrim(str)); 
    % catch str = ''; continue;
    % end;

    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end


    try 
        j = vocabList.(str);
        word_indices(end+1) = j;
    catch
        continue;
    end


end

end
