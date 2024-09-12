import os

def clear_outputs(folder_name):
    output_path = f'output/{folder_name}/'
    if os.path.exists(output_path):
        for file in os.listdir(output_path):
            os.remove(output_path + file)
    else:
        print(f'Folder {folder_name} does not exist')