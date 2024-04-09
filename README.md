Contains the code for a simple microservice for beat detection. Because the installation is a bit complicated and I'd rather not disturb the main repo with it, I've put it in a separate repo. The code is not very clean, but it works. I'll clean it up later, unless I am not bothered enough.

0. Use Python 3.10. Start a new conda env
1. Install ffmpeg
2. pip install spleeter
3. pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
4. pip install Cython mido
5. git clone --recursive https://github.com/CPJKU/madmom.git && mv madmom tmp && mv tmp/* . && rm -rf tmp. If you suspect that the madmom installation is broken, you can install from commit hash 0551aa8, which is known to work
6. python setup.py develop --user
7. git clone --branch=main https://github.com/zhaojw1998/Beat-Transformer
8. pip install uvicorn==0.22.0
9. pip install fastapi==0.95.1
10. pip install librosa==0.10.1
11. Run everything else
12. Pray

If you are inside the main FYP repo, run this, will be easier
```bash
pip install spleeter
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install Cython mido
python setup.py develop --user
pip install librosa==0.10.1
```
