from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from loguru import logger


class LSFEMS(object):
    COLUMNS = [
        "FEMS_1ST_ELEC_ACB_17_HZ",
        "FEMS_1ST_ELEC_ACB_17_KW",
        "FEMS_1ST_ELEC_ACB_17_KWH",
        "FEMS_1ST_ELEC_ACB_17_PF",
        "FEMS_1ST_ELEC_ACB_18_HZ",
        "FEMS_1ST_ELEC_ACB_18_KW",
        "FEMS_1ST_ELEC_ACB_18_KWH",
        "FEMS_1ST_ELEC_ACB_18_PF",
        "FEMS_1ST_ELEC_ACB_19_1_MCCB02_HZ",
        "FEMS_1ST_ELEC_ACB_19_1_MCCB02_KW",
        "FEMS_1ST_ELEC_ACB_19_1_MCCB02_KWH",
        "FEMS_1ST_ELEC_ACB_19_1_MCCB02_PF",
        "MES_ENERGY_HEAT_HEAT_2",
        "FEMS_1ST_ELEC_ACB_17_KWH_DIFF",
        "FEMS_1ST_ELEC_ACB_18_KWH_DIFF",
        "FEMS_1ST_ELEC_ACB_19_1_MCCB02_KWH_DIFF",
    ]

    def __init__(
        self,
        win_size,
        step,
        label_column,
        train_data_path="",
        test_data_path="",
        val_data_path="",
        mode="train",
    ):

        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.train_data_path = (train_data_path,)
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.scaler = MinMaxScaler()

        train_data = pd.read_csv(train_data_path).ffill().bfill()
        train_data.columns = train_data.columns.str.replace(".", "_")
        self.train = train_data[LSFEMS.COLUMNS].values

        self.scaler.fit(self.train)
        self.train = self.scaler.transform(self.train)
        logger.info(f"train: {self.train.shape}")

        if mode == "train":
            return

        val_data = pd.read_csv(val_data_path).ffill().bfill()
        test_data = pd.read_csv(test_data_path).ffill().bfill()

        self.val = val_data[LSFEMS.COLUMNS].values
        self.test = test_data[LSFEMS.COLUMNS].values

        self.val_labels = val_data[label_column].values
        self.test_labels = test_data[label_column].values

        self.val = self.scaler.transform(self.val)
        self.test = self.scaler.transform(self.test)

        logger.info(f"val: {self.val.shape}")
        logger.info(f"test: {self.test.shape}")
        logger.info(f"val_labels: {self.val_labels.shape}")
        logger.info(f"test_labels: {self.test_labels.shape}")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index : index + self.win_size]), 0.0
        elif self.mode == "val":
            return np.float32(self.val[index : index + self.win_size]), np.float32(
                self.val_labels[index : index + self.win_size]
            )
        elif self.mode == "test":
            return np.float32(self.test[index : index + self.win_size]), np.float32(
                self.test_labels[index : index + self.win_size]
            )
        else:  # self.mode == 'thre' 슬라이딩 윈도우
            return (
                np.float32(
                    self.test[
                        index
                        // self.step
                        * self.win_size : index
                        // self.step
                        * self.win_size
                        + self.win_size
                    ]
                ),
                np.float32(
                    self.test_labels[
                        index
                        // self.step
                        * self.win_size : index
                        // self.step
                        * self.win_size
                        + self.win_size
                    ]
                ),
            )


def get_loader_segment(
    train_data_path,
    val_data_path,
    test_data_path,
    batch_size,
    label_column,
    win_size=100,
    step=100,
    mode="train",
    dataset="KDD",
):

    if dataset == "LSFEMS":
        dataset = LSFEMS(
            win_size=win_size,
            step=step,
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            val_data_path=val_data_path,
            mode=mode,
            label_column=label_column
        )

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return data_loader
