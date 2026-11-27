FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRAUDSHIELD_DATA=/app/data/transactions.csv.gz \
    FRAUDSHIELD_NEXT_MONTH=/app/data/next_month.csv.gz \
    FRAUDSHIELD_REGISTRY=/app/models/registry \
    FRAUDSHIELD_REPORTS=/app/reports

WORKDIR /app
RUN useradd --create-home --uid 10001 fraudshield

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY data ./data
# Bake a trained model + drift report into the image so the container is useful on first start.
RUN fraudshield train && fraudshield drift && chown -R fraudshield:fraudshield /app

USER fraudshield
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "fraudshield.api:app", "--host", "0.0.0.0", "--port", "8000"]
