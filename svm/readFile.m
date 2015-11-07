function file_contents = readFile(fileName)

    fid = fopen(fileName);
    if fid
        file_contents = fscanf(fid, '%c', inf);
        fclose(fid);
    else
        file_contents = '';
        fprintf('Unable to open %s\n', fileName);
    end

end

