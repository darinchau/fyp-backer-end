from .beat_pred import beats_prediction
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

def create_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.post('/beat')
    async def root(r: Request):
        audio_data = await r.body()
        data = beats_prediction(audio_data)
        return data

    @app.get('/alive')
    async def alive():
        return {"alive": "true"}
