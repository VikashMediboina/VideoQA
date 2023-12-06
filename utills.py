import os

def remove_file(file_path):
    if os.path.exists(file_path):
    # Delete the file
        os.remove(file_path)
        print(f"File {file_path} has been deleted.")
    else:
        print(f"The file {file_path} does not exist.")
def remove_allFiles(output_folder):
    if os.path.exists(output_folder):
        # If it does, remove all existing videos in the folder
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")