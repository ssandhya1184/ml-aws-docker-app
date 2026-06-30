FROM python:3.11-slim

WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install --no-cache-dir uv && uv sync

# Copy rest of the app
COPY . .

# Fix imports
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8501

# Run app
#CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uv", "run", "streamlit","run", "src/app/streamlit_app.py", "--server.address=0.0.0.0",  "--server.port=8501"]