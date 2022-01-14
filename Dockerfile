# Base image

FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy files to docker
COPY test_environment.py test_environment.py
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY Makefile Makefile
COPY sweep.yaml sweep.yaml
COPY tests/ tests/
COPY entrypoints.py entrypoints.py

RUN mkdir models

# Install ML_things and google.cloud storage
RUN apt update
RUN apt install -y git
RUN apt install -y wget
RUN rm -rf /var/lib/apt/lists/*
RUN pip3 install git+https://github.com/gmihaila/ml_things.git --no-cache-dir
RUN pip3 install google-cloud-storage --no-cache-dir
RUN pip3 install google-cloud-secret-manager --no-cache-dir

# Install requirements
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
    --path-update=false --bash-completion=false \
    --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["python", "-u", "entrypoints.py"]

