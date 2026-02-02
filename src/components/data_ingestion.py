import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# ================= CONFIG CLASS =================
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "data.csv")


# ================= MAIN CLASS =================
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered Data Ingestion Component")

        try:
            # ===== Read Dataset =====
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Dataset read successfully")

            # ===== Create Artifact Folder =====
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # ===== Save Raw Data =====
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw Data Saved")

            # ===== Train Test Split =====
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # ===== Save Train Test =====
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Train Test Split Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# ================= RUN FILE =================
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    print("Train File Saved At:", train_path)
    print("Test File Saved At:", test_path)
