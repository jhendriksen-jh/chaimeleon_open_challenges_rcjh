# Builds pytorch dev conntainer
FROM pytorch/pytorch:latest

# install pip
RUN python3 -m ensurepip --upgrade

# install required packages
COPY ./../../requirements.txt ./
COPY ./../../requirements-dev.txt ./
RUN pip install --upgrade pip wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-dev.txt

ENV PYTHONPATH=.

WORKDIR .