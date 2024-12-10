"""The dataloader for Palmvein dataset.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoad import BaseLoad
from tqdm import tqdm

class PalmVeinLoader(BaseLoad):
    """The data loader for the Palmvein dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an palmvein dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- gt_HR.csv
                             |-- gt_SpO2.csv
                             |-- wave.csv
                             |-- hr_gt.csv
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- gt_HR.csv
                             |-- gt_SpO2.csv
                             |-- wave.csv
                             |-- hr_gt.csv
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- gt_HR.csv
                             |-- gt_SpO2.csv
                             |-- wave.csv
                             |-- hr_gt.csv
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path (For Palmvein dataset)."""
        # 获取 subject 文件夹
        data_dirs = glob.glob(data_path + os.sep + "subject*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = [{"index": re.search('subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""
        if begin == 0 and end == 1:
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = [data_dirs[i] for i in choose_range]
        return data_dirs_new

    def preprocess_dataset(self, data_dirs, config_preprocess, _i, _file_list_dict):
        """Preprocesses the raw data."""            
        file_num = len(data_dirs)
        for i in range(file_num):
            # Read preprocessed images (hand region images)
            image_folder = os.path.join(data_dirs[i]["path"], "images")
            frames = self.read_images_from_folder(image_folder)

            # Read Labels
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(os.path.join(data_dirs[i]['path'], "wave.csv"))

            target_length = frames.shape[0]
            bvps = BaseLoad.resample_ppg(bvps, target_length)
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)         
            self.preprocessed_data_len += self.save(frames_clips, bvps_clips, data_dirs[i]["index"])
     

    @staticmethod
    def read_csv(file_path):
        """Reads a CSV file and returns its content."""
        return pd.read_csv(file_path) 

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)
    # def read_video(video_file):
    #     """Reads a video file, returns frames(T, H, W, 3) """
    #     VidObj = cv2.VideoCapture(video_file)
    #     VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    #     success, frame = VidObj.read()
    #     frames = list()
    #     while success:
    #         frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
    #         frame = np.asarray(frame)
    #         frames.append(frame)
    #         success, frame = VidObj.read()
    #     return np.asarray(frames)

    def read_wave(self, bvp_file):
        """Reads a bvp signal file."""
        data = pd.read_csv(bvp_file)
        # waves = data['HR'].values  # 假设列名是 'Wave'
        waves = data['Wave'].values  # 假设列名是 'Wave'
        return np.asarray(waves)