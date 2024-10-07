
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os 

class LSDummyX(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # 'x_normal.csv' 파일 로드 (train data)
        train_data = pd.read_csv(os.path.join(data_path, 'x_normal.csv'))
        
        # 'x_abnormal.csv' 파일 로드 (val and test data)
        abnormal_data = pd.read_csv(os.path.join(data_path, 'x_abnormal.csv'))
        
        # 필요한 컬럼 선택
        feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']
        label_column = ['is_abnormal']
        
        # 'x_abnormal.csv'를 5:5로 분할하여 val_data와 test_data 생성
        val_data, test_data = train_test_split(abnormal_data, test_size=0.5, random_state=42)
        
        # 특성과 레이블 분리
        self.train = train_data[feature_columns].values
        self.val = val_data[feature_columns].values
        self.test = test_data[feature_columns].values
        
        self.val_labels = val_data[label_column].values
        self.test_labels = test_data[label_column].values
        
        # NaN 값을 0으로 대체
        self.train = np.nan_to_num(self.train)
        self.val = np.nan_to_num(self.val)
        self.test = np.nan_to_num(self.test)
        
        # 데이터 스케일링 (train 데이터로 fit)
        self.scaler.fit(self.train)
        self.train = self.scaler.transform(self.train)
        self.val = self.scaler.transform(self.val)
        self.test = self.scaler.transform(self.test)
        
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("val_labels:", self.val_labels.shape)
        print("test_labels:", self.test_labels.shape)
        
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), 0.0
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else: # self.mode == 'thre' 슬라이딩 윈도우
            return (np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), 
                    np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]))


class LSDummyY(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # 'x_normal.csv' 파일 로드 (train data)
        train_data = pd.read_csv(os.path.join(data_path, 'y_normal.csv'))
        
        # 'x_abnormal.csv' 파일 로드 (val and test data)
        abnormal_data = pd.read_csv(os.path.join(data_path, 'y_abnormal.csv'))
        
        # 필요한 컬럼 선택
        feature_columns = ['y1', 'y2', 'y3']
        label_column = ['is_abnormal']
        
        # 'x_abnormal.csv'를 5:5로 분할하여 val_data와 test_data 생성
        val_data, test_data = train_test_split(abnormal_data, test_size=0.5, random_state=42)
        
        # 특성과 레이블 분리
        self.train = train_data[feature_columns].values
        self.val = val_data[feature_columns].values
        self.test = test_data[feature_columns].values
        
        self.val_labels = val_data[label_column].values
        self.test_labels = test_data[label_column].values
        
        # NaN 값을 0으로 대체
        self.train = np.nan_to_num(self.train)
        self.val = np.nan_to_num(self.val)
        self.test = np.nan_to_num(self.test)
        
        # 데이터 스케일링 (train 데이터로 fit)
        self.scaler.fit(self.train)
        self.train = self.scaler.transform(self.train)
        self.val = self.scaler.transform(self.val)
        self.test = self.scaler.transform(self.test)
        
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("val_labels:", self.val_labels.shape)
        print("test_labels:", self.test_labels.shape)
        
    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), 0.0
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[index:index + self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else: # self.mode == 'thre' 슬라이딩 윈도우
            return (np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), 
                    np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]))

def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    if (dataset == 'LSDummyY'):
        dataset = LSDummyY(data_path, win_size, step, mode)
    elif (dataset == 'LSDummyX'):
        dataset = LSDummyX(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
