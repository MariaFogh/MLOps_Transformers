# Base image

FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY test_environment.py test_environment.py
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY entrypoints.sh entrypoints.sh
COPY src/ src/
COPY data/ data/
COPY Makefile Makefile
COPY sweep.yaml sweep.yaml
COPY tests/ tests/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

#ENTRYPOINT ["python", "-u", "src/data/predict_model.py"]
ENTRYPOINT ["./entrypoints.sh"]
