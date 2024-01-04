import backoff
import dagshub
import shutil
import os
from dagshub.data_engine import datasources
from dotenv import load_dotenv

load_dotenv()


DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO = os.environ.get("DAGSHUB_REPO")

DAGSHUB_FULL_REPO= DAGSHUB_REPO_OWNER+"/"+DAGSHUB_REPO

ds = datasources.get(DAGSHUB_FULL_REPO, os.environ.get("DATASOURCE_NAME"))

data_length = len(ds)
# Assuming ds contains a list of dictionaries where each dictionary has 'image_name' and 'class_name'
# Adjust these keys based on your actual dataset structure
target_classes = ['d-solide', 'd-liquide']

# Filter data based on the specified target classes
filtered_data = ds.filter(lambda item: item['class_name'] in target_classes)

# Create directories for train, valid, and test
train_dir = "train"
valid_dir = "valid"
test_dir = "test"

os.mkdir(train_dir, exist_ok=True)
os.mkdir(valid_dir, exist_ok=True)
os.mkdir(test_dir, exist_ok=True)

# Organize data into folders based on class names
for class_name in target_classes:
    class_train_dir = os.path.join(train_dir, class_name)
    class_valid_dir = os.path.join(valid_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)

    os.mkdir(class_train_dir, exist_ok=True)
    os.mkdir(class_valid_dir, exist_ok=True)
    os.mkdir(class_test_dir, exist_ok=True)

# Split data into train, valid, and test sets and copy images to corresponding folders
for item in filtered_data:
    image_name = item['image_name']
    class_name = item['class_name']

    # Decide whether the image goes to train, valid, or test
    # You can use a more sophisticated splitting logic here
    if some_condition_for_train:
        shutil.copy(image_name, os.path.join(train_dir, class_name))
    elif some_condition_for_valid:
        shutil.copy(image_name, os.path.join(valid_dir, class_name))
    elif some_condition_for_test:
        shutil.copy(image_name, os.path.join(test_dir, class_name))

# Note: Adjust 'some_condition_for_train', 'some_condition_for_valid', 'some_condition_for_test'
# based on your actual splitting logic. It could be a random split, fixed percentages, etc.
