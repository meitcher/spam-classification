function fileList = getAllFiles(dirName)
  dirData = readdir(dirName);
  fileList = char(dirData(3:end));
end