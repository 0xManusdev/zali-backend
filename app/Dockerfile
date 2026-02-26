FROM python:3.11-slim

# Install system deps for pillow and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy all app code (depuis le dossier app)
COPY . /app

# Ensure uploads dir exists
RUN mkdir -p /app/uploads

# Expose port (Render injecte $PORT at runtime)
ENV PORT=8000

# Start script to allow $PORT
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000
ENTRYPOINT ["/start.sh"]