# build stage
ARG BASE_IMAGE=python:3.12-slim
FROM ubuntu:latest as builder
WORKDIR /src/

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    libgomp1

# runtime stage
FROM $BASE_IMAGE as runtime-environment

#FROM python:3.12-slim

ENV PYTHONUNBUFFERED True

# set the working directory
WORKDIR /usr/src/app

# install libgomp1 agagin
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1

# install dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH=/usr/src/app/src

# copy src code
COPY ./src ./src

EXPOSE 8080

# start the server
#CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4006", "--proxy-headers"]

# start the server
CMD ["streamlit", "run", "src/main.py","--server.port=8080","--server.address=0.0.0.0"]