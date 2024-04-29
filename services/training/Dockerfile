# Use the official PyTorch image as the base image
FROM pytorch/pytorch:latest

WORKDIR /app

COPY . /app


# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Specify CUDA compatibility (adjust based on your CUDA version)
ENV TORCH_CUDA_ARCH_LIST "7.5"

CMD [ "python3 code/zenml_running.py" ]

