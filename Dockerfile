FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch and dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    sentencepiece \
    flask

# Clone DeepSeek model
RUN git clone https://github.com/deepseek-ai/deepseek-coder.git

ENV MODEL_PATH=/app/deepseek-model
ENV HF_HOME=/app/huggingface_cache

# Download model
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-7b-base'); \
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-7b-base'); \
    model.save_pretrained('${MODEL_PATH}'); \
    tokenizer.save_pretrained('${MODEL_PATH}')"

COPY inference.py .

EXPOSE 8080

CMD ["python", "inference.py"]
