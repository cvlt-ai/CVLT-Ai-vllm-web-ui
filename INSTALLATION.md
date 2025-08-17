# Installation Guide - vLLM Gradio WebUI

This guide will walk you through the complete installation and setup process for the vLLM Gradio WebUI.

##  Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **Python**: 3.8 or higher (3.10+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (recommended but not required)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB+ free space for models and data

### CUDA Setup (For GPU Support)
If you have an NVIDIA GPU, install CUDA:

#### Ubuntu/Linux:
```bash
# Install CUDA (example for CUDA 11.8)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### Windows:
Download and install CUDA from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

##  Installation Steps

### Step 1: Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd vllm-gradio-webui

# Or extract the downloaded ZIP file
unzip vllm-gradio-webui.zip
cd vllm-gradio-webui
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note**: This may take 10-15 minutes depending on your internet connection and system.

### Step 4: Install Optional Dependencies

#### For OCR Support (Text extraction from images):
```bash
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract

# Windows:
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### For Advanced Image Processing:
```bash
pip install opencv-python
```

### Step 5: Verify Installation
```bash
# Run the basic test script
python test_basic.py
```

You should see output like:
```
==================================================
vLLM Gradio WebUI - Basic Tests
==================================================
Testing imports...
✓ ApplicationManager imported
✓ GradioInterface imported
...
Test Results: 4/4 passed
✓ All tests passed! The application should work correctly.
```

##  First Run

### Basic Startup
```bash
python main.py
```

The application will start and be available at: `http://localhost:7860`

### With Custom Settings
```bash
# Start on different port
python main.py --port 8080

# Enable debug mode
python main.py --debug

# Multi-GPU setup (if you have multiple GPUs)
python main.py --tensor-parallel-size 2

# Start with specific host (for remote access)
python main.py --host 0.0.0.0 --port 7860
```

##  Setting Up Models

### Option 1: Download Models Manually
1. Create a models directory:
   ```bash
   mkdir -p models
   ```

2. Download models from Hugging Face:
   ```bash
   # Example: Download a small model for testing
   git lfs install
   git clone https://huggingface.co/microsoft/DialoGPT-medium models/DialoGPT-medium
   ```

### Option 2: Use the Web Interface
1. Start the application
2. Go to the "Models" tab
3. Enter model paths or Hugging Face model names
4. Click "Scan Models" or "Load Model"



##  Configuration

### Basic Configuration
The application uses `config/app_config.yaml` for configuration. Key settings:

```yaml
# Model settings
vllm:
  model_dirs:
    - "./models"  # Add your model directories here
  gpu:
    tensor_parallel_size: 1  # Number of GPUs for tensor parallelism
    gpu_memory_utilization: 0.9  # GPU memory usage (0.0-1.0)

# RAG settings
rag:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 200

# Web scraping
web:
  enabled: true
  scraping_timeout: 30

# File processing
files:
  enabled: true
  max_file_size: 52428800  # 50MB
```

### Environment Variables
Create a `.env` file for sensitive settings:
```bash
# Optional: OpenAI API key for embeddings
OPENAI_API_KEY=your_openai_api_key

# Optional: Google Search API
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
```

##  Troubleshooting

### Common Issues and Solutions

#### 1. "CUDA out of memory" Error
```bash
# Reduce GPU memory utilization
python main.py --gpu-memory-utilization 0.7

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
python main.py
```

#### 2. Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 3. Port Already in Use
```bash
# Use different port
python main.py --port 8080

# Or find and kill the process using the port
lsof -ti:7860 | xargs kill -9  # Linux/macOS
```

#### 4. Model Loading Fails
- Check model path exists
- Verify sufficient disk space
- Ensure model format is compatible with vLLM
- Check GPU memory availability

#### 5. Web Interface Not Loading
- Check firewall settings
- Verify the correct URL (http://localhost:7860)
- Try different browser
- Check console for JavaScript errors

### Debug Mode
For detailed troubleshooting:
```bash
python main.py --debug --log-level DEBUG
```

### Getting Help
1. Check the logs in `vllm_gradio_webui.log`
2. Run the test script: `python test_basic.py`
3. Check system requirements and dependencies
4. Review the troubleshooting section in README.md

##  Performance Optimization

### For Better Performance:
1. **Use SSD storage** for models and data
2. **Increase GPU memory utilization** if stable
3. **Use tensor parallelism** for multiple GPUs
4. **Enable model quantization** for memory efficiency
5. **Optimize chunk sizes** for RAG based on your documents

### Multi-GPU Setup:
```bash
# 2 GPUs with tensor parallelism
python main.py --tensor-parallel-size 2

# 4 GPUs with pipeline parallelism
python main.py --pipeline-parallel-size 4

# Check GPU usage
nvidia-smi
```

## 📊Monitoring

### System Monitoring
The web interface includes:
- GPU usage and memory
- System performance metrics
- Model loading status
- RAG document statistics
- Application health checks

### Log Files
- `vllm_gradio_webui.log` - Main application log
- `logs/` directory - Component-specific logs

##  Updates and Maintenance

### Updating Dependencies
```bash
# Activate virtual environment
source venv/bin/activate

# Update all packages
pip install -r requirements.txt --upgrade

# Or update specific packages
pip install --upgrade vllm gradio torch
```

### Backing Up Data
Important directories to backup:
- `data/` - Uploaded files and processed content
- `config/` - Configuration files
- `models/` - Downloaded models (if stored locally)

##  You're Ready!

Your vLLM Gradio WebUI is now installed and ready to use. Start the application with:

```bash
python main.py
```

Then open your browser to `http://localhost:7860` and enjoy your advanced LLM interface!

For more detailed usage instructions, see the main README.md file.

