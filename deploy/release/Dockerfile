# Builds pytorch dev conntainer
FROM pytorch/pytorch:latest


#open-challenge things
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output
    
USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

# install pip
RUN python3 -m ensurepip --upgrade
RUN python3 -m pip install --user -U pip && python3 -m pip install --user pip-tools

# install required packages
COPY --chown=user:user ./../../requirements.txt /opt/app/
RUN pip install --upgrade pip wheel
RUN python3 -m piptools sync requirements.txt
RUN pip install SimpleITK evalutils

COPY --chown=user:user ./../../library/process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]