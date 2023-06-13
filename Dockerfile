# We start from a PyTorch 1.10 image with CUDA 11.1 and Python 3.8
FROM pytorch/pytorch:1.10.0-cuda11.1-cudnn8-runtime

# Install required libraries
RUN pip install pytorch-lightning torchvision

# Copy your Python scripts into the Docker image
COPY . /workspace
WORKDIR /workspace

# The command that will be run when the container is started
CMD ["python", "your_training_script.py"]
