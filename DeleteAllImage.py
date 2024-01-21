import os

folderpath = "output/"

files = os.listdir(folderpath)

for file in files:
    if file.lower().endswith("png"):
        file_path = os.path.join(folderpath, file)
        os.remove(file_path)
        