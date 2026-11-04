FROM python:3.12-slim

# Install system dependencies needed for XGBoost and OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run container as user 1000 (standard for Hugging Face Spaces & cloud security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Cache dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (backend, datasets, frontend, models)
COPY --chown=user . .
RUN chmod +x start.sh

# Default port: 7860 (Hugging Face Spaces) or overridden by $PORT on Render
EXPOSE 7860
CMD ["./start.sh"]
