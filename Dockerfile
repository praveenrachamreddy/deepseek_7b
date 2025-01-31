FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    sentencepiece

# Clone DeepSeek model
RUN git clone https://github.com/deepseek-ai/deepseek-coder.git

# Set environment variables
ENV MODEL_PATH=/app/deepseek-model
ENV HF_HOME=/app/huggingface_cache

# Download model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-7b-base'); \
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-7b-base'); \
    model.save_pretrained('${MODEL_PATH}'); \
    tokenizer.save_pretrained('${MODEL_PATH}')"

# Copy inference script
COPY inference.py .

# Expose port for potential API
EXPOSE 8080

# Run inference script
CMD ["python", "inference.py"]
