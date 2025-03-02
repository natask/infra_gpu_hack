 # Start from the offical Python 3.9 base image.
 # This will include all the basic components we need.
 FROM python:3.10
 
 # Within the container, we'll work in this directory.
 WORKDIR /code
 
 # Copy our requirements.txt file into the container.
 COPY ./requirements.txt requirements.txt
 
 # Install our requirements within the container.
 RUN pip install --no-cache-dir --upgrade -r requirements.txt
 
 # Copy our source code directory into the container.
 COPY . .
 
 # Expose port 80 within our container to the outside world.
 EXPOSE 80
 
 # Specify the command to run when the container is started.
 # In this case, start our FastAPI server on port 80.
 CMD ["python", "main.py"]