import os

rotation_estimation_folder = "C:\\Users\\paulb\\Documents\\TUDresden\\Bachelor\\datasets\\rotation_estimation"

#clear the rotation estimation folder
def clear_rotation_dataset():
    for dir in os.listdir(rotation_estimation_folder):
        dir_path_1 = os.path.join(rotation_estimation_folder, dir)
        for dir in os.listdir(dir_path_1):
            dir_path_2 = os.path.join(dir_path_1, dir)
            for file in os.listdir(dir_path_2):
                file_path = os.path.join(dir_path_2, file)
                print(f"Removing {file_path}")
                os.remove(file_path)
    print("Rotation dataset cleared.")

clear_rotation_dataset()