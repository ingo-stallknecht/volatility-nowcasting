# Minimal, robust Dockerfile for Streamlit on Python 3.11
FROM python:3.11-slim

# Workdir
WORKDIR /app

# Ensure both common pip bin paths are on PATH (prevents "streamlit: not found")
ENV PATH="/root/.local/bin:/usr/local/bin:${PATH}"

# Install runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code and demo data
COPY app.py ./app.py
COPY data ./data

# Streamlit defaults
ENV PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Start the app
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
