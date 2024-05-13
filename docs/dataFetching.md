# Data fetching with dagshub
Code snippet in dagshub_data_load.py downloads and organizes data from a CSV file and DagsHub repository for machine learning model training in the Map-Action-Model repository architecture.

**Environment Variables**:
   - `DAGSHUB_REPO_OWNER`: Owner of the repository.
   - `DAGSHUB_REPO`: Repository name.
   - `DATASOURCE_NAME`: Name of the datasource configured in DagsHub.

The dataset is divided as follows:
- **Training Data**: 70%
- **Validation Data**: 20%
- **Test Data**: 10%


::: steps.dagshub_utils.dagshub_data_load