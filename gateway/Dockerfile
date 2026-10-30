# Gateway image. Small and always-on -- it's meant to run on a cheap CPU
# node pool, not the GPU pool, so it must stay cheap to schedule.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
RUN pip install .

EXPOSE 8080

# Non-root.
RUN useradd -u 10001 -m gateway
USER 10001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
