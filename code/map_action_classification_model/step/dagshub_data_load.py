import os
import pandas as pd
import numpy as np
import requests
from glob import glob
from dotenv import load_dotenv
from dagshub.data_engine import datasources
from zenml.steps import step, Output, BaseStepConfig

load_dotenv()

DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO")

DAGSHUB_FULL_REPO = f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO}"

@step
def download_and_organize_data() -> Output(
    data_dir = str,
    train_dir = str,
    valid_dir = str,
    test_dir = str
):
    # Load data from a CSV file
    ds = datasources.get(DAGSHUB_FULL_REPO, os.environ.get("DATASOURCE_NAME"))
    ds = ds.all().dataframe
    df = pd.read_csv("project-7.csv", usecols=["choice", "image"])
    df_img = df["image"]
    ds_img = ds["dagshub_download_url"]

    # Create directories for train, valid, and test
    data_dir = "data"
    train_dir = "train"
    valid_dir = "valid"
    test_dir = "test"

    # Randomly permute indices
    data_volumes = rand_shuffle = np.random.permutation(df_img.shape[0])
    print(data_volumes)

    # Download images and organize them into directories
    for n in range(df_img.shape[0]):
        for m in range(ds_img.shape[0]):
            if df_img[n] == ds_img[m]:
                save_dir = os.path.join(data_dir, df['choice'][n])
                try:
                    os.makedirs(save_dir)
                except FileExistsError:
                    pass  # Directory already exists
                response = requests.get(ds_img[m], stream=True)
                with open(f"{save_dir}/images{n+1}.jpeg", 'wb') as file:
                    for chunk in response.iter_content(chunk_size=128):
                        file.write(chunk)
                print(f"File downloaded successfully in {n+1}")

    # Gather file paths
    types = ('*.jpg', '*.jpeg', '*.png')
    files = [f for ext in types for f in glob(os.path.join(data_dir, f'*/{ext}'))]
    print(files)

    # Organize files into train, valid, and test directories
    for i, split in enumerate(['valid', 'test', 'test']):
        for n in data_volumes[i * 20: (i + 1) * 20]:
            folder = files[n].split('/')[1]
            name = files[n].split('/')[-1]
            try:
                os.makedirs(os.path.join(data_dir, split, folder))
            except FileExistsError:
                pass  # Directory already exists
            os.rename(files[n], os.path.join(data_dir, split, folder, name))

    return {
        'data_dir': os.path.abspath(data_dir),
        'train_dir': os.path.abspath(train_dir),
        'valid_dir': os.path.abspath(valid_dir),
        'test_dir': os.path.abspath(test_dir),
    }





