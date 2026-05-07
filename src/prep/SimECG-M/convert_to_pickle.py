import os
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DEFAULT_ROOT_DIR = os.path.join(
    REPO_ROOT, "outputs", "experiment", "v250312", "simulator", "ecgsyn", "250403-122531"
)

def dat_to_np(dat_file: str, target_length: int):
    """

    Args:
        dat_file (str): _description_
    """

    data = np.loadtxt(dat_file, usecols=(1,))

    # if longer than target_length, truncate
    if len(data) > target_length:
        data = data[:target_length]

    return data

def dat_dir_to_pickle(dat_dir: str, savename: str, target_length: int):
    """

    Args:
        dat_dir (str): _description_
    """
    # Get all directories in the dat_dir
    data_list = []
    subdirs = glob(os.path.join(dat_dir, "id*"))
    for subdir in tqdm(subdirs):
        # Get all .dat files in the subdir
        dat_files = glob(os.path.join(subdir, "*.dat"))

        # Read each .dat file and append to the list
        for dat_file in dat_files:
            data = dat_to_np(dat_file, target_length)
            data_list.append(data)

    data_list = np.array(data_list)

    # Save the list to a pickle file
    with open(savename, 'wb') as f:
        pickle.dump(data_list, f)

def process_split(dat_dirs: list, save_dir: str, target_length: int):
    """
    Split the dat_dirs into train and val directories
    Args:
        dat_dirs (list): _description_
        save_root (str): _description_
    """
    for i, dat_dir in enumerate(dat_dirs):
        print(f"Processing {i+1}/{len(dat_dirs)}: {dat_dir}")
        savename = os.path.join(save_dir, f"idx{i+1:06d}.pkl")
        dat_dir_to_pickle(dat_dir, savename, target_length)

def prepare_pickle(root_dir: str, target_length: int):
    """
    Prepare pickle files from dat files

    Args:
        root_dir (str): _description_
    """
    # Set.
    dat_dir = os.path.join(root_dir, "dat_files")
    save_root = os.path.join(root_dir, "pickle_files")

    # Split.
    dat_dirs = glob(os.path.join(dat_dir, "id*"))
    train_dirs, val_dirs = train_test_split(
        dat_dirs, test_size=0.1, random_state=42)
    
    # Train set.
    save_dir_train = os.path.join(save_root, "train/samples")
    os.makedirs(save_dir_train, exist_ok=True)
    process_split(train_dirs, save_dir_train, target_length)
    
    # Validation set.
    save_dir_val = os.path.join(save_root, "val/samples")
    os.makedirs(save_dir_val, exist_ok=True)
    process_split(val_dirs, save_dir_val, target_length)
    print("Done")

def id_to_filename(id_num: int, dat_dir: str):
    """
    Get the directory of the id

    Args:
        id_num (int): _description_
        dat_dir (str): _description_
    """
    target_file = os.path.join(
        dat_dir, 
        f"id{id_num//1000+1:04d}", 
        f"id{id_num:08d}",
        f"syn{id_num:08d}.dat"
    )
    if os.path.exists(target_file):
        return target_file
    return None

if __name__ == "__main__":

    # dat_file = "{}/dat_files/id0001/id00000002/syn00000002.dat".format(
    #     DEFAULT_ROOT_DIR
    # )
    # root = DEFAULT_ROOT_DIR

    # dat_file = id_to_filename(0, root)    
    # data = dat_to_np(dat_file)
    # print(data)

    # dat_file = id_to_filename(1, root)
    # data = dat_to_np(dat_file)
    # print(data)

    # dat_file = id_to_filename(717395, root)
    # data = dat_to_np(dat_file)
    # print(data)

    # dat_file = id_to_filename(717396, root)
    # data = dat_to_np(dat_file)
    # print(data)

    # dat_file = id_to_filename(717397, root)
    # data = dat_to_np(dat_file)
    # print(data)

    root = DEFAULT_ROOT_DIR
    target_length = 5000 # 500Hz x 10s
    prepare_pickle(root, target_length)

    # dat_dir = os.path.join(
    #     REPO_ROOT, "outputs", "experiment", "v250312", "simulator", "ecgsyn",
    #     "250402-215240", "dat_files", "id0871"
    # )
    # save_dir = os.path.join(
    #     REPO_ROOT, "outputs", "experiment", "v250312", "simulator", "ecgsyn",
    #     "250402-215240", "pickle_files"
    # )
    # os.makedirs(save_dir, exist_ok=True)
    # dat_dir_to_pickle(dat_dir, save_dir)
