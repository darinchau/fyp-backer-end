FROM python:3.10

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Install spleeter
RUN pip install spleeter

# Install PyTorch and its dependencies
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install Cython and mido
RUN pip install Cython mido

# Clone and install madmom
RUN pip install -q -U git+https://github.com/CPJKU/madmom.git

# Clone Beat-Transformer
RUN git clone --branch=main https://github.com/zhaojw1998/Beat-Transformer

# Install additional dependencies
RUN pip install uvicorn==0.22.0 fastapi==0.95.1 librosa==0.10.1 gunicorn==21.2.0
