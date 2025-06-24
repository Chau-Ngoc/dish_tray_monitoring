FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./train.py ./monitor.py ./
RUN pip install -r requirements.txt

RUN mkdir -p /app/data /app/models /app/runs

ENTRYPOINT ["python"]
