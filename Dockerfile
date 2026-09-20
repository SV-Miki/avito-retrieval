FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.14.0+cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY pyproject.toml .

CMD ["python", "-m", "src.submission", "--output", "answer.csv"]
