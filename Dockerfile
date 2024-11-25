# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app into the container
COPY app.py .

# Copy the model into the container
COPY outputs/best_model.pkl outputs/best_model.pkl
COPY outputs/dropped_columns.pkl outputs/dropped_columns.pkl
COPY outputs/dv.joblib outputs/dv.joblib

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]