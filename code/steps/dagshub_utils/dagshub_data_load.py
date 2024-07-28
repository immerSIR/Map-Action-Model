import os
import pandas as pd
import numpy as np
import requests
import torch
from glob import glob
from dotenv import load_dotenv
from dagshub.data_engine import datasources
from typing import Annotated, Optional, Tuple
from zenml import step, pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Verify environment variables
required_env_vars = ["DAGSHUB_REPO_OWNER",
                     "DAGSHUB_REPO", "DATASOURCE_NAME", "DAGSHUB_TOKEN"]
for var in required_env_vars:
    if not os.environ.get(var):
        raise EnvironmentError(
            f"Required environment variable {var} is not set")

DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO")
DAGSHUB_FULL_REPO = f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}"


@step
def download_and_organize_data() -> Tuple[
    Annotated[str, "train_dir"],
    Annotated[str, "valid_dir"],
    Annotated[str, "test_dir"],
    Annotated[int, "batch_size"],
]:
    """
    Download and organize data from a CSV file and DagsHub repository.
    Returns:
    Tuple[str, str, str, int]: Directories for train, valid, and test, and batch size.
    """
    try:
        logger.info(
            f"Attempting to fetch data from DagsHub repo: {DAGSHUB_FULL_REPO}")
        ds = datasources.get(
            DAGSHUB_FULL_REPO, os.environ.get("DATASOURCE_NAME"))
        ds = ds.all().dataframe
        logger.info(
            f"Successfully fetched data from DagsHub. Shape: {ds.shape}")
    except Exception as e:
        logger.error(f"Error fetching data from DagsHub: {str(e)}")
        raise

    try:
        df = pd.read_csv("project_9.csv", usecols=["choice", "image"])
        logger.info(f"Successfully read project_9.csv. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error("project_9.csv not found")
        raise
    except Exception as e:
        logger.error(f"Error reading project_9.csv: {str(e)}")
        raise

    df_img = df["image"]
    ds_img = ds["path"]

    # Create directories for train, valid, and test
    data_dir = "data"
    train_dir = "train"
    valid_dir = "valid"
    test_dir = "test"
    os.makedirs(data_dir, exist_ok=True)

    # Randomly permute indices
    data_volumes = np.random.permutation(df_img.shape[0])
    logger.info(f"Number of images to process: {df_img.shape[0]}")

    # Download images and organize them into directories
    for n in range(df_img.shape[0]):
        for m in range(ds_img.shape[0]):
            if df_img[n].split('/')[-1] == ds_img[m]:
                save_dir = os.path.join(data_dir, df['choice'][n])
                os.makedirs(save_dir, exist_ok=True)
                try:
                    response = requests.get(df_img[n], stream=True)
                    response.raise_for_status()  # Raises an HTTPError for bad responses
                    with open(f"{save_dir}/images{n+1}.jpg", 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    logger.info(f"File {n+1} downloaded successfully")
                except requests.RequestException as e:
                    logger.error(f"Error downloading file {n+1}: {str(e)}")

    # Gather file paths
    types = ('*.jpg', '*.jpeg', '*.png')
    files = []
    for ext in types:
        files.extend(glob(os.path.join(data_dir, '*/' + ext)))
    logger.info(f"Total files found: {len(files)}")

    # Organize files into train, valid, and test directories
    test_split = 30
    valid_split = int(0.15 * len(files))  # 15% for validation

    for n in data_volumes[:test_split]:
        folder = files[n].split('/')[1]
        name = files[n].split('/')[-1]
        os.makedirs(os.path.join(data_dir, test_dir, folder), exist_ok=True)
        os.rename(files[n], os.path.join(data_dir, test_dir, folder, name))

    for n in data_volumes[test_split:test_split+valid_split]:
        folder = files[n].split('/')[1]
        name = files[n].split('/')[-1]
        os.makedirs(os.path.join(data_dir, valid_dir, folder), exist_ok=True)
        os.rename(files[n], os.path.join(data_dir, valid_dir, folder, name))

    for n in data_volumes[test_split+valid_split:]:
        folder = files[n].split('/')[1]
        name = files[n].split('/')[-1]
        os.makedirs(os.path.join(data_dir, train_dir, folder), exist_ok=True)
        os.rename(files[n], os.path.join(data_dir, train_dir, folder, name))

    train_dir = f"{data_dir}/{train_dir}"
    valid_dir = f"{data_dir}/{valid_dir}"
    test_dir = f"{data_dir}/{test_dir}"
    batch_size = 20

    logger.info(
        f"Data organization complete. Train dir: {train_dir}, Valid dir: {valid_dir}, Test dir: {test_dir}")
    logger.info(f"Train samples: {len(glob(os.path.join(train_dir, '*/*')))}")
    logger.info(
        f"Validation samples: {len(glob(os.path.join(valid_dir, '*/*')))}")
    logger.info(f"Test samples: {len(glob(os.path.join(test_dir, '*/*')))}")

    return train_dir, valid_dir, test_dir, batch_size
