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
RUN git clone --recursive https://github.com/CPJKU/madmom.git && \
    mv madmom tmp && \
    mv tmp/* . && \
    rm -rf tmp && \
    python setup.py develop --user

# Clone Beat-Transformer
RUN git clone --branch=main https://github.com/zhaojw1998/Beat-Transformer

# Install additional dependencies
RUN pip install uvicorn==0.22.0 fastapi==0.95.1 librosa==0.10.1

# Set the working directory
WORKDIR /app

# Copy the app files to the container
COPY . /app

# Move the files
RUN mv ./Beat-Transformer/code/DilatedTransformer.py ./DilatedTransformer.py
RUN mv ./Beat-Transformer/code/DilatedTransformerLayer.py ./DilatedTransformerLayer.py

# Run the app
CMD ["python", "app.py"]
