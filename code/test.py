import os
import shutil
from pathlib import Path

# Move all files from subfolders to parent folder
data_path = Path("/home/julian/Documents/Spezialisierung/SPEZ/data/animal_data")

for file_path in data_path.rglob("*"):
    if file_path.is_file() and file_path.parent != data_path:
        shutil.move(str(file_path), str(data_path / file_path.name))

# Remove empty subdirectories
for dir_path in data_path.iterdir():
    if dir_path.is_dir():
        try:
            dir_path.rmdir()  # Only removes if empty
        except OSError:
            pass  # Directory not empty, skip