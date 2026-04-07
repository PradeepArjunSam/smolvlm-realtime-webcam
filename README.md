# SmolVLM Real-time Camera Demo

Real-time webcam / IP camera feed analysis using **SmolVLM 500M** and a local inference server.

Supports **llama.cpp**, **Ollama**, and **vLLM** backends.

---

## Quick Start

### 1. Start the inference server

**llama.cpp (default, recommended)**
```bash
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF
# Add -ngl 99 to use GPU (NVIDIA / AMD / Intel)
```

**Ollama**
```bash
ollama run smolvlm
```

**vLLM**
```bash
vllm serve HuggingFaceTB/SmolVLM-500M-Instruct
```

> You can try other multimodal models from the [llama.cpp multimodal docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).

---

### 2a. Browser-based client (index.html)

1. Open `index.html` in your browser (Chrome / Edge recommended)
2. Select your **backend** (llama.cpp / Ollama / vLLM)
3. Select your **camera** from the dropdown
4. Optionally tweak the **instruction**
5. Click **Start** and enjoy!

> Works on `localhost` or any **HTTPS** origin.  
> For JSON output: include `{}` or the word `json` in your instruction.

---

### 2b. Python OpenCV client (client.py)

Install dependencies:
```bash
pip install -r requirements.txt
```

**llama.cpp (default)**
```bash
python client.py
```

**Ollama**
```bash
python client.py --backend ollama --url http://localhost:11434 --model smolvlm
```

**vLLM**
```bash
python client.py --backend vllm --url http://localhost:8000 --model HuggingFaceTB/SmolVLM-500M-Instruct
```

**IP Camera / Phone (using "IP Webcam" Android app)**
```bash
python client.py --camera http://192.168.1.100:8080/video
```

**All options**
```
python client.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--url` | `http://localhost:8080` | Server base URL |
| `--backend` | `llamacpp` | `llamacpp`, `ollama`, `vllm`, `custom` |
| `--model` | _(empty)_ | Model name (required for Ollama/vLLM) |
| `--camera` | `0` | Camera index or HTTP/RTSP URL |
| `--interval` | `0.5` | Seconds between requests |
| `--max-tokens` | `150` | Max response length |
| `--timeout` | `10.0` | Request timeout in seconds |
| `--prompt` | _(default)_ | Custom instruction |
| `--no-mirror` | false | Disable horizontal flip |

**Keyboard shortcuts** (Python client):
- `q` — quit
- `p` — pause / resume requests

---

## CPU-only Usage

SmolVLM 500M is small enough to run on CPU, though it will be slow:
```bash
llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF
# Omit -ngl flag to run on CPU only
```

For Apple Silicon use the Metal build of llama.cpp — `-ngl 99` enables full GPU offload.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Failed to parse messages: Unsupported content part type` | Update llama.cpp to the latest version |
| JSON output doesn't work | Add `{}` or the word `json` to your instruction |
| Camera permission denied | Use HTTPS or `localhost` |
| Request too slow | Increase the interval or reduce `--max-tokens` |
| Connection refused | Ensure server is running and URL matches |

---

## Credits

- Original demo by [ngxson](https://github.com/ngxson/smolvlm-realtime-webcam)
- Fork with Python client, Ollama/vLLM support and bug fixes by [PradeepArjunSam](https://github.com/PradeepArjunSam/smolvlm-realtime-webcam)
- Model: [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) by HuggingFace
