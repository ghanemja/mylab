function n = count_file_words(filename)
%COUNT_FILE_WORDS Number of whitespace-separated words in a file.
txt = fileread(filename);
n = numel(regexp(txt, '[^ \t\r\n]+', 'match'));
end
