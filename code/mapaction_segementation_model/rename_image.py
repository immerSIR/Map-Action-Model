import os

def rename_files(directory):
    i = 0
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)

        # Modify this part based on how you want to rename your files
        new_filename = f"D-Solide{i}"
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} to {new_filename}")
        i+=1

# Replace 'your_directory_path' with the path of the directory containing the files you want to rename
rename_files('/home/mapaction/Documents/Exp-data/Mask/D-Solide/')