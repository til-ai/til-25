# using base python image bc RL agents don't often use much gpu
FROM --platform=linux/amd64 python:3.11-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# pip gives a warning if you install packages as root
# set this flag to just ignore the warning
ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install -U pip
WORKDIR /workspace

# install other requirements
COPY docker-requirements.txt .
RUN pip install -r docker-requirements.txt

# copy the rest of the files into the container
COPY src .

# start model service
CMD uvicorn api_service:app --port 5000 --host 0.0.0.0