FROM ubuntu:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    transformers \
    torch \
    numpy \
    pandas \
    accelerate

# Create working directory
WORKDIR /app

# Copy all necessary files
COPY main.py .
COPY generate.py .

# Run the application
CMD ["python3", "llada_main.py"]

