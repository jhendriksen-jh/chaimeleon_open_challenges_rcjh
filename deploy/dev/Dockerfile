# Builds pytorch dev conntainer
FROM pytorch/pytorch:latest

# install pip
RUN python3 -m ensurepip --upgrade

# install required packages
COPY ./../../requirements.txt ./
RUN pip install --upgrade pip wheel
RUN pip install --upgrade -r requirements.txt

ENV PYTHONPATH=.

WORKDIR .