import pytest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from code.steps.dagshub_utils.dagshub_data_load import download_and_organize_data


@pytest.fixture
def mock_os():
    with patch('os.makedirs') as mocked_makedirs, \
            patch('os.path.join', side_effect=lambda *args: '/'.join(args)) as mocked_join:
        yield mocked_makedirs, mocked_join


@pytest.fixture
def mock_requests():
    with patch('requests.get') as mocked_get:
        mocked_response = MagicMock()
        mocked_response.iter_content.return_value = [b'data']
        mocked_get.return_value = mocked_response
        yield mocked_get


@pytest.fixture
def mock_glob():
    with patch('glob.glob') as mocked_glob:
        def glob_side_effect(pattern):
            if '*/' in pattern:
                return ['data/train/train_image1.jpg', 'data/test/test_image1.jpg']
            return []
        mocked_glob.side_effect = glob_side_effect
        yield mocked_glob


@pytest.fixture
def mock_pandas_read_csv():
    with patch('pandas.read_csv') as mocked_read_csv:
        mocked_df = pd.DataFrame(
            {'choice': ['train', 'test'], 'image': ['url1', 'url2']})
        mocked_read_csv.return_value = mocked_df
        yield mocked_read_csv


@pytest.fixture
def mock_datasources():
    with patch('dagshub.data_engine.datasources.get') as mocked_get:
        mocked_ds = MagicMock()
        mocked_ds.all().dataframe = pd.DataFrame(
            {'path': ['train_image1.jpg', 'test_image1.jpg']})
        mocked_get.return_value = mocked_ds
        yield mocked_get


def test_download_and_organize_data(mock_os, mock_requests, mock_glob, mock_pandas_read_csv, mock_datasources):
    try:
        train_dir, valid_dir, test_dir, batch_size = download_and_organize_data()
    except IndexError as e:
        print(e)
        raise

    # Assertions
    assert train_dir == 'data/train'
    assert valid_dir == 'data/valid'
    assert test_dir == 'data/test'
    assert batch_size == 20

    mocked_makedirs, mocked_join = mock_os
    mocked_makedirs.assert_called()  # Called to create directories
    mocked_join.assert_called()  # Called to join paths
    mock_requests.assert_called()  # Called to make HTTP requests
    mock_glob.assert_called()  # Called to get file paths
    mock_pandas_read_csv.assert_called_once_with(
        "project-8.csv", usecols=['choice', 'image'])
    mock_datasources.assert_called()  # Called to get datasources


if __name__ == '__main__':
    pytest.main()
