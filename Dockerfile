FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages 
RUN pip install --no-cache-dir -r app/requirements.txt

# Use a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run detect_objects.py when the container launches
ENTRYPOINT ["python", "main.py"]