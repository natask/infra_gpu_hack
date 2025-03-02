# Serve Speculate

An OpenAI-compatible API server using FastAPI and vLLM for serving large language models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The server will start on port 7050.

## API Endpoints

### Chat Completions
- Endpoint: `/v1/chat/completions`
- Method: POST
- Compatible with OpenAI chat completion format

Example request:
```bash
curl http://localhost:7050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## Configuration

The default model is set to "facebook/opt-125m". You can modify the model in `main.py` by changing the `LLM` initialization.
