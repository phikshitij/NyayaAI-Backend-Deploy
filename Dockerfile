FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Hugging Face Spaces requires a non-root user with ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set up environment variables for Hugging Face cache
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the actual application files
COPY --chown=user . $HOME/app

# Hugging Face Spaces automatically routes traffic to port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
