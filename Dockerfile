FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    transformers==4.49.0 \
    torch==2.1.2 \
    numpy==1.26.3 \
    pandas==2.1.4 \
    accelerate==0.26.1 \
    datasets==2.13.0 \
    modelscope==1.9.5 \
    transformers_stream_generator==0.0.4 \
    flask \
    flask-cors

# Create working directory and copy files
RUN pip3 install pandas
WORKDIR /app

# Copy all necessary files
COPY llada_main.py .
COPY generate.py .
COPY generate_vanilla_prm.py .
COPY math_test_data.csv .
COPY generate_backmasking.py .
COPY llada_main_bon.py . 


# Run the application
CMD ["python3", "llada_main_bon.py"]
