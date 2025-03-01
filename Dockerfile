FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    transformers==4.36.2 \
    torch==2.1.2 \
    numpy==1.26.3 \
    pandas==2.1.4 \
    accelerate==0.26.1 \
    datasets==2.16.1

# Create working directory and copy files
WORKDIR /app

# Copy all necessary files
COPY llada_main.py .
COPY generate.py .
COPY prompts.txt .


# Run the application
CMD ["python3", "llada_main.py"]

