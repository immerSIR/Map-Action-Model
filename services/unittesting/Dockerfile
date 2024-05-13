# Use an official Python runtime as a parent image
FROM python:3.10.13-bookworm

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install virtualenv

RUN virtualenv env

RUN source env/bin/activate

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

# Run linting with flake8

RUN pip install flake8

# Set the working directory to /app/MapAction
# Install coverage and run tests
RUN pip install -r requirements.txt
RUN pytest -n auto --asyncio-mode=auto --cov=. --cov-report=xml --cov-report=term-missing
