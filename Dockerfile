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
    pandas

# Copy and run the application
COPY main.py .
CMD ["python3", "main.py"]

