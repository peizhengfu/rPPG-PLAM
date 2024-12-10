"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import csv
import glob
import os
import re
import time
from math import ceil
from scipy import signal
from scipy import sparse
from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
import math
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from torch.utils.data import Dataset
from tqdm import tqdm

class BaseLoad(Dataset):
    """The base class for data loading based on pytorch Dataset.

    The dataloader supports both providing data for pytorch training and common data-preprocessing methods,
    including reading files, resizing each frame, chunking, and video-signal synchronization.
    """

    @staticmethod
    def add_data_loader_args(parser):
        """Adds arguments to parser for training process"""
        parser.add_argument(
            "--cached_path", default=None, type=str)
        parser.add_argument(
            "--preprocess", default=None, action='store_true')
        return parser

    def __init__(self, dataset_name, raw_data_path, config_data):
        """Inits dataloader with lists of files.

        Args:
            dataset_name(str): name of the dataloader.
            raw_data_path(string): path to the folder containing all data.
            config_data(CfgNode): data settings(ref:config.py).
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.inputs = list()
        self.labels = list()
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        self.cached_path = config_data.CACHED_PATH
        self.file_list_path = config_data.FILE_LIST_PATH
        self.preprocessed_data_len = 0
        self.data_format = config_data.DATA_FORMAT
        self.do_preprocess = config_data.DO_PREPROCESS
        self.config_data = config_data

        assert (config_data.BEGIN < config_data.END)
        assert (config_data.BEGIN > 0 or config_data.BEGIN == 0)
        assert (config_data.END < 1 or config_data.END == 1)
        if config_data.DO_PREPROCESS:
            self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
            self.preprocess_dataset(self.raw_data_dirs, config_data.PREPROCESS, config_data.BEGIN, config_data.END)
        else:
            if not os.path.exists(self.cached_path):
                print('CACHED_PATH:', self.cached_path)
                raise ValueError(self.dataset_name,
                                 'Please set DO_PREPROCESS to True. Preprocessed directory does not exist!')
            if not os.path.exists(self.file_list_path):
                print('File list does not exist... generating now...')
                self.raw_data_dirs = self.get_raw_data(self.raw_data_path)
                self.build_file_list_retroactive(self.raw_data_dirs, config_data.BEGIN, config_data.END)
                print('File list generated.', end='\n\n')

            self.load_preprocessed_data()
        print('Cached Data Path', self.cached_path, end='\n\n')
        print('File List Path', self.file_list_path)
        print(f" {self.dataset_name} Preprocessed Dataset Length: {self.preprocessed_data_len}", end='\n\n')

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.inputs)
    

    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and it's corresponding signals(T)."""
        data = np.load(self.inputs[index])
        label = np.load(self.labels[index])
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')
        data = np.float32(data)
        label = np.float32(label)
        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]
        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]
        return data, label, filename, chunk_id

    def get_raw_data(self, raw_data_path):
        """Returns raw data directories under the path.

        Args:
            raw_data_path(str): a list of video_files.
        """
        raise Exception("'get_raw_data' Not Implemented")

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        raise Exception("'split_raw_data' Not Implemented")

    def read_npy_video(self, video_file):
        """Reads a video file in the numpy format (.npy), returns frames(T,H,W,3)"""
        frames = np.load(video_file[0])
        if np.issubdtype(frames.dtype, np.integer) and np.min(frames) >= 0 and np.max(frames) <= 255:
            processed_frames = [frame.astype(np.uint8)[..., :3] for frame in frames]
        elif np.issubdtype(frames.dtype, np.floating) and np.min(frames) >= 0.0 and np.max(frames) <= 1.0:
            processed_frames = [(np.round(frame * 255)).astype(np.uint8)[..., :3] for frame in frames]
        else:
            raise Exception(f'Loaded frames are of an incorrect type or range of values! '\
            + f'Received frames of type {frames.dtype} and range {np.min(frames)} to {np.max(frames)}.')
        return np.asarray(processed_frames)

    def generate_pos_psuedo_labels(self, frames, fs=30):
        """Generated POS-based PPG Psuedo Labels For Training

        Args:
            frames(List[array]): a video frames.
            fs(int or float): Sampling rate of video
        Returns:
            env_norm_bvp: Hilbert envlope normalized POS PPG signal, filtered are HR frequency
        """

        # generate POS PPG signal
        WinSec = 1.6
        RGB = POS_WANG._process_video(frames)
        N = RGB.shape[0]
        H = np.zeros((1, N))
        l = math.ceil(WinSec * fs)

        for n in range(N):
            m = n - l
            if m >= 0:
                Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
                Cn = np.mat(Cn).H
                S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
                h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
                mean_h = np.mean(h)
                for temp in range(h.shape[1]):
                    h[0, temp] = h[0, temp] - mean_h
                H[0, m:n] = H[0, m:n] + (h[0])

        bvp = H
        bvp = utils.detrend(np.mat(bvp).H, 100)
        bvp = np.asarray(np.transpose(bvp))[0]

        # filter POS PPG w/ 2nd order butterworth filter (around HR freq)
        # min freq of 0.7Hz was experimentally found to work better than 0.75Hz
        min_freq = 0.70
        max_freq = 3
        b, a = signal.butter(2, [(min_freq) / fs * 2, (max_freq) / fs * 2], btype='bandpass')
        pos_bvp = signal.filtfilt(b, a, bvp.astype(np.double))

        # apply hilbert normalization to normalize PPG amplitude
        analytic_signal = signal.hilbert(pos_bvp) 
        amplitude_envelope = np.abs(analytic_signal) # derive envelope signal
        env_norm_bvp = pos_bvp/amplitude_envelope # normalize by env

        return np.array(env_norm_bvp) # return POS psuedo labels
    
    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """Parses and preprocesses all the raw data based on split.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        """
        data_dirs_split = self.split_raw_data(data_dirs, begin, end)  # partition dataset 
        print("Data directories after split:", data_dirs_split)  # 打印划分后的数据路径
        # send data directories to be processed
        file_list_dict = self.multi_process_manager(data_dirs_split, config_preprocess) 
        self.build_file_list(file_list_dict)  # build file list
        self.load_preprocessed_data()  # 加载所有数据及对应标签
        print("Total Number of raw files preprocessed:", len(data_dirs), end='\n\n')

    def preprocess(self, frames, bvps, config_preprocess):
        """Preprocesses a pair of data (frames and bvps)."""
        
        # 由于图片已经裁剪好，直接使用 resize
        frames = self.resize_images(frames, config_preprocess.RESIZE.W, config_preprocess.RESIZE.H)
        
        # Check data transformation type
        data = list()  # Video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoad.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoad.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoad.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoad.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = self.chunk(data, bvps, config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])
            bvps_clips = self.standardized_label(bvps_clips)
        return frames_clips, bvps_clips
        
    def resize_images(frames, width, height):
            """Resize frames to specified width and height."""
            resized_frames = np.zeros((frames.shape[0], height, width, 3), dtype=frames.dtype)
            for i in range(frames.shape[0]):
                resized_frames[i] = cv2.resize(frames[i], (width, height), interpolation=cv2.INTER_AREA)
            return resized_frames
    
    # def hand_detection(self, frame, use_larger_box=False, larger_box_coef=1.5):
    #     """Hand detection on a single frame.

    #     Args:
    #         frame(np.array): a single frame.
    #         use_larger_box(bool): whether to use a larger bounding box on hand detection.
    #         larger_box_coef(float): coefficient for the larger box.
    #         roi_size(tuple): size of the ROI to be extracted.
    #     Returns:
    #         roi_box_coor(list[int]): coordinates of resized ROI bounding box in the form [x_min, y_min, x_max, y_max].
    #     """
    #     # 添加延迟，等待模型和视频流初始化
    #     # time.sleep(2)
    #     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     image.flags.writeable = True
    #     results = self.hands.process(image)
        
    #     hand_box_coor = None
    #     if results.multi_hand_landmarks:
    #         print(f"Detected {len(results.multi_hand_landmarks)} hands")
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             hand_landmarks = results.multi_hand_landmarks[0]  # 只取第一个检测到的手掌
    #             # Extract landmarks and calculate bounding box
    #             h, w, _ = frame.shape
    #             landmark_arr = np.array([(landmark.x, landmark.y) for landmark in hand_landmarks.landmark])
    #             selected_landmarks = landmark_arr[[0, 5, 9, 13, 17], :]
    #             x_min, y_min = np.min(selected_landmarks, axis=0)
    #             x_max, y_max = np.max(selected_landmarks, axis=0)
                
    #             # 扩大边界框大小
    #             padding = 20
    #             x_min = max(int(x_min * w) - padding, 0)
    #             x_max = min(int(x_max * w) + padding, w)
    #             y_min = max(int(y_min * h) - padding, 0)
    #             y_max = min(int(y_max * h) + padding, h)
    #             # hand_box_coor = [x_min, y_min, x_max - x_min, y_max - y_min]
    #             # hand_box_coor = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    #             hand_box_coor = [int(x_min), int(y_min), int(x_max), int(y_max)]
    #             # break
    #             # 提取手掌区域并保存为图像
    #             hand_roi = frame[y_min:y_max, x_min:x_max]  # 提取手掌的ROI
    #             if hand_roi.size > 0:
    #                 cv2.imwrite("detected_hand.png", hand_roi)  # 保存手掌图像
    #                 print("Saved detected hand image as 'detected_hand.png'.")

    #             break  # 只处理第一个手掌
            
    #     if hand_box_coor is None:
    #         print("ERROR: No Hand Detected")
    #         hand_box_coor = [0, 0, frame.shape[1], frame.shape[0]]
    #     print(hand_box_coor)
    #     return hand_box_coor   

    # def crop_hand_resize(self, frames, use_larger_box=True, larger_box_coef=1.2, width=256, height=256):
    #     """Detect hand keypoints and return bounding box coordinates for each frame and resize the frames.

    #     Args:
    #         frames(np.array): Video frames.
    #         use_larger_box(bool): Whether enlarge the detected bounding box from hand detection.
    #         larger_box_coef(float): The coefficient of the larger region (height and width),
    #                                 the middle point of the detected region will stay still during the process of enlarging.
    #         width(int): The width of the resized frame.
    #         height(int): The height of the resized frame.
    #     Returns:
    #         resized_frames(np.array): Array of resized frames.
    #     """
    #     resized_frames = np.zeros((frames.shape[0], height, width, 3), dtype=frames.dtype)
    #     for i in range(frames.shape[0]):
    #         frame = frames[i]
    #         # Detect the hand region
    #         x_min, y_min, x_max, y_max = self.hand_detection(frame)
    #         # Optionally enlarge the bounding box
    #         if use_larger_box:
    #             center_x = (x_min + x_max) / 2
    #             center_y = (y_min + y_max) / 2
    #             box_w = (x_max - x_min) * larger_box_coef
    #             box_h = (y_max - y_min) * larger_box_coef
    #             x_min = int(center_x - box_w / 2)
    #             y_min = int(center_y - box_h / 2)
    #             x_max = int(center_x + box_w / 2)
    #             y_max = int(center_y + box_h / 2)

    #             # Ensure the bounding box is within frame boundaries
    #             x_min = max(0, x_min)
    #             y_min = max(0, y_min)
    #             x_max = min(frame.shape[1], x_max)
    #             y_max = min(frame.shape[0], y_max)
    #         # Crop the frame based on the bounding box
    #         cropped_frame = frame[y_min:y_max, x_min:x_max]
    #         print(f"Original frame shape: {frame.shape}")
    #         print(f"Cropped frame shape: {cropped_frame.shape}")
    #         # Resize the cropped frame
    #         resized_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_AREA)
    #         print(f"Resized frame shape: {resized_frame.shape}")
    #         resized_frames[i] = resized_frame
    #     return resized_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of hand cropped frames
            bvp_clips: all chunks of bvp frames
        """

        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)


    def save(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data.

        Args:
            frames_clips(np.array): chunks of hand cropped frames.
            bvps_clips(np.array): chunks of blood volumne pulse (PPG) labels.
            filename: name of the file.
        Returns:
            count: count of preprocessed data.
        """

        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            self.inputs.append(input_path_name)
            self.labels.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return count


    def save_multi_process(self, frames_clips, bvps_clips, filename):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(bvps_clips)):
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=8):
        """Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
            multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs)
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()  # multi-process manager
        file_list_dict = manager.dict()  # dictionary for all processes to store processed files
        p_list = []  # list of processes
        running_num = 0  # number of running processes

        # in range of number of files to process
        for i in choose_range:
            process_flag = True
            while process_flag:  # ensure that every i creates a process
                if running_num < multi_process_quota:  # in case of too many processes
                    # send data to be preprocessing task
                    p = Process(target=self.preprocess_dataset_subprocess, 
                                args=(data_dirs,config_preprocess, i, file_list_dict))
                    p.start()
                    p_list.append(p)
                    running_num += 1
                    process_flag = False
                for p_ in p_list:
                    if not p_.is_alive():
                        p_list.remove(p_)
                        p_.join()
                        running_num -= 1
                        pbar.update(1)
        # join all processes
        for p_ in p_list:
            p_.join()
            pbar.update(1)
        pbar.close()

        return file_list_dict

    def build_file_list(self, file_list_dict):
        """Build a list of files used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """
        file_list = []
        # iterate through processes and add all processed file paths
        for _, file_paths in file_list_dict.items():
            file_list = file_list + file_paths

        if not file_list:
            raise ValueError(self.dataset_name, 'No files in file list')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """ If a file list has not already been generated for a specific data split build a list of files 
        used by the dataloader for the data split. Eg. list of files used for 
        train / val / test. Also saves the list to a .csv file.

        Args:
            data_dirs(List[str]): a list of video_files.
            begin(float): index of begining during train/val split.
            end(float): index of ending during train/val split.
        Returns:
            None (this function does save a file-list .csv file to self.file_list_path)
        """

        # get data split based on begin and end indices.
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)

        # generate a list of unique raw-data file names
        filename_list = []
        for i in range(len(data_dirs_subset)):
            filename_list.append(data_dirs_subset[i]['index'])
        filename_list = list(set(filename_list))  # ensure all indexes are unique

        # generate a list of all preprocessed / chunked data files
        file_list = []
        for fname in filename_list:
            processed_file_data = list(glob.glob(self.cached_path + os.sep + "{0}_input*.npy".format(fname)))
            file_list += processed_file_data

        if not file_list:
            raise ValueError(self.dataset_name,
                             'File list empty. Check preprocessed data folder exists and is not empty.')

        file_list_df = pd.DataFrame(file_list, columns=['input_files'])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)  # save file list to .csv

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list.

        Args:
            None
        Returns:
            None
        """
        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs = file_list_df['input_files'].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len - 1):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)
