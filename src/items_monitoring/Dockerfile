FROM ultralytics/ultralytics:8.3.159-cpu

WORKDIR /app

COPY ./requirements.txt ./train.py ./monitor.py ./
RUN pip install -r requirements.txt

RUN mkdir -p /app/data /app/models /app/runs

ENTRYPOINT ["python"]
