FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg && \
    wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb && \
    dpkg -i cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends cudnn-cuda-12 && \
    rm -rf /var/lib/apt/lists/* /var/cudnn-local-repo-ubuntu2204-9.1.1*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY ./etc /app/etc

EXPOSE 8000
