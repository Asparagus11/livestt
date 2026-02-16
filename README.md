# LiveSTT - Speech-to-Text with Whisper

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)

A modern, multilingual speech-to-text application powered by OpenAI's Whisper model with real-time transcription, LLM-based text correction, and a user-friendly web interface.

## ✨ Features

- 🎤 **Real-time Transcription** - Live audio recording with instant transcription
- 📁 **File Upload** - Support for WAV, MP3, M4A, OGG, FLAC formats
- 🌍 **Multilingual** - UI in German, English, French + 12+ transcription languages
- 🤖 **LLM Correction** - Automatic grammar and punctuation correction (Ollama/OpenAI)
- ⚡ **GPU Acceleration** - Automatic CUDA detection for faster processing
- 💾 **Archive Management** - Save, view, and manage transcriptions
- 🎛️ **Advanced Settings** - Fine-tune Whisper parameters (VAD, beam size, temperature)
- 🔒 **Secure** - Path traversal protection, file size limits, input sanitization

## 📋 Requirements

- Python 3.8 or higher
- 2GB RAM minimum (4GB+ recommended)
- Optional: NVIDIA GPU with CUDA for acceleration
- Optional: Ollama or OpenAI API for text correction

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/live-stt.git
cd live-stt
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) GPU Support

For NVIDIA GPUs with CUDA:
```bash
pip install nvidia-cublas-cu12  # For CUDA 12.x
# or
pip install nvidia-cublas-cu11  # For CUDA 11.x
```

The application automatically detects and uses available GPUs.

## ⚙️ Configuration

Edit `config.py` to customize settings:

```python
# Server
PORT = 8003

# Whisper Model
WHISPER_MODEL = "base"  # tiny, base, small, medium, large

# Device (auto-detects GPU)
DEVICE = "auto"  # "auto", "cpu", "cuda"
COMPUTE_TYPE = "auto"  # "auto", "int8", "float16"

# Upload Limits
MAX_UPLOAD_SIZE_MB = 100  # 0 = unlimited

# LLM Provider
LLM_PROVIDER = "ollama"  # "ollama" or "openai"
LLM_MODEL = "llama3.2"
LLM_BASE_URL = "http://localhost:11434"

# For OpenAI
OPENAI_API_KEY = ""  # Your API key
OPENAI_MODEL = "gpt-4"
```

## 🎯 Usage

### Start the Server
```bash
python main.py
```

Open your browser: **http://localhost:8003**

### Real-time Transcription
1. Click "Start Recording"
2. Allow microphone access
3. Speak - transcription appears in real-time
4. Click "Stop Recording"
5. Optionally correct with LLM
6. Save with custom filename suffix

### File Upload
1. Click "Choose File" in upload section
2. Select audio file (WAV, MP3, M4A, OGG, FLAC)
3. Click "Transcribe File"
4. Wait for transcription to complete

### Settings
- **Whisper Model**: Choose speed vs. quality (tiny → medium)
- **Language**: Select audio language (auto-detect or specific)
- **Advanced Settings**: Fine-tune VAD, beam size, temperature

## 🌍 LLM Integration

### Ollama (Local, Free)
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Set in `config.py`: `LLM_PROVIDER = "ollama"`

### OpenAI (Cloud, Paid)
1. Get API key: https://platform.openai.com
2. Set in `config.py`:
   ```python
   LLM_PROVIDER = "openai"
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4"
   ```

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch
```

### GPU not detected
- Check CUDA installation: `nvidia-smi`
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall: `pip install nvidia-cublas-cu12`

### Ollama connection error
- Start Ollama: `ollama serve`
- Check URL in config: `LLM_BASE_URL = "http://localhost:11434"`

### File upload fails
- Check file size limit in `config.py`: `MAX_UPLOAD_SIZE_MB`
- Ensure file format is supported

### Poor transcription quality
- Use larger model: `WHISPER_MODEL = "medium"`
- Adjust VAD threshold in Advanced Settings
- Ensure clear audio (low background noise)

## 📊 Model Comparison

| Model  | Size  | Speed (CPU) | Speed (GPU) | Quality |
|--------|-------|-------------|-------------|---------|
| tiny   | 40MB  | ~2x realtime | ~10x realtime | ⭐⭐ |
| base   | 150MB | ~1x realtime | ~8x realtime | ⭐⭐⭐ |
| small  | 500MB | ~0.5x realtime | ~5x realtime | ⭐⭐⭐⭐ |
| medium | 1.5GB | ~0.2x realtime | ~3x realtime | ⭐⭐⭐⭐⭐ |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Ollama](https://ollama.ai) - Local LLM runtime

## 📧 Support

For issues and questions, please use the [GitHub Issues](https://github.com/yourusername/live-stt/issues) page.

---

Made with ❤️ by the LiveSTT community

