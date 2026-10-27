# Demo API. Build from the repo root: docker build -f app/Dockerfile -t resolvr-api .
# Needs the bundle from app/build_index.py in app/data/ and the pipeline's work/{retriever,dicts.json} mounted at /work.
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && grep -vE "^(torch|--extra)" requirements.txt | pip install --no-cache-dir -r /dev/stdin
COPY src/ src/
COPY app/ app/
COPY results/ results/
ENV BER_WORK=/work BER_ROOT=/srv
EXPOSE 7400
CMD ["uvicorn", "server:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "7400"]
