FROM python:3.10-slim-bookworm

WORKDIR /app

COPY ./requirements.txt ./app.py ./
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
