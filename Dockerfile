# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /test
WORKDIR C:\gp\job_description_manipulation\two_features_modelllll.joblib
# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI code into the container
COPY . .

# Expose the port your FastAPI app is running on (default is 8000)
EXPOSE 8888

# Start the FastAPI server
CMD ["uvicorn", "test:app", "--host", "0.0.0.0", "--port", "8888"]
