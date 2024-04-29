# Setup and Running Process

## Overview

This section outlines the steps required to set up and run the continuous training pipeline for the machine learning model using Docker Compose. The process involves configuring environment variables and utilizing a Docker Compose file to automate the deployment and operation of the training environment.

## Environment Setup

### Configuring Environment Variables

Before launching the training pipeline, it is crucial to set up the required environment variables. These variables are essential for configuring the application's settings and ensuring the correct operation of the Docker containers.

`
Create an `.env` file in the root directory of your project with the following structure:
`


# Environment configuration
`
DAGSHUB_REPO_OWNER=your_dagshub_username
DAGSHUB_REPO=your_repository_name
DATASOURCE_NAME=your_datasource_name
`

## Running the Training Pipeline

To run the continuous training pipeline, use the Docker Compose file specifically prepared for this purpose. The Docker Compose configuration manages the necessary services, volumes, and networks to facilitate the training process.

### Starting the Pipeline

Run the following command in your terminal to start the training pipeline:

```
docker-compose -f _ct.yml up
```