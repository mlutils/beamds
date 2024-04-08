# ARG instruction to define the base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# ARG instructions for other parameters
ARG REQUIREMENTS_FILE
ARG ALGORITHM_DIR
ARG ENTRYPOINT_SCRIPT

# Copy the requirements.txt file and install Python dependencies
COPY ${REQUIREMENTS_FILE} /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the algorithm bundle directory
COPY ${ALGORITHM_DIR} /app/algorithm

# Copy the entry point script
COPY ${ENTRYPOINT_SCRIPT} /app/entrypoint.py

# Set the working directory
WORKDIR /app

# Set the entry point script as the entry point for the Docker container
ENTRYPOINT ["python", "/app/entrypoint.py"]