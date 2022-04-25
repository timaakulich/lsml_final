from typing import Any

import uvicorn
from celery.result import AsyncResult
from fastapi import FastAPI, UploadFile, status, File, Body, HTTPException, Request  # noqa
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.nn_model import mlflow_client, get_task_state
from backend.tasks import train_nn_model_task, generate_lyrics_task, celery_app
import pandas as pd
import io


app = FastAPI(title='ML Project')
templates = Jinja2Templates(directory='backend/templates')


class TrainSchema(BaseModel):
    artist: str
    size: int
    task_id: str


class PredictSchema(BaseModel):
    task_id: str


class TaskStatusSchema(BaseModel):
    ready: bool
    data: Any
    state: dict = None


class ArtistsList(BaseModel):
    __root__ = list[str]


@app.post('/train', status_code=status.HTTP_202_ACCEPTED, response_model=TrainSchema)  # noqa
async def train_model_view(
        file: UploadFile = File(
            ...,
            description='CSV: [artist].csv. Columns: lyrics'
        ),
        artist: str = Body(..., embed=True),
        epochs: int = Body(20, embed=True, gt=0)
):
    try:
        text = (await file.read()).decode()
        df = pd.read_csv(io.StringIO(text))  # noqa
        text = df['lyrics'].str.cat(sep='\n').lower()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Invalid input file! Must contain \'lyrics\' column'
        )
    task = train_nn_model_task.delay(artist, text, epochs)
    return {
        'artist': artist,
        'size': len(text),
        'task_id': task.id
    }


@app.post('/predict', status_code=status.HTTP_202_ACCEPTED, response_model=PredictSchema)   # noqa
async def predict_lyrics_view(
        artist: str = Body(..., embed=True),
        start_text: str = Body(..., embed=True, min_length=10),
        length: int = Body(400, embed=True, ge=10),
):
    artists = set(map(lambda x: x.name, mlflow_client.list_registered_models()))  # noqa
    if artist not in artists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Artist not found'
        )
    task = generate_lyrics_task.delay(artist, start_text.lower(), length)
    return {
        'task_id': task.id,
    }


@app.get('/task/{task_id}', response_model=TaskStatusSchema)
async def task_status_view(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    return {
        'ready': task.ready(),
        'state': get_task_state(task.id),
        'data': task.result if task.ready() else None
    }


@app.get('/artists', include_in_schema=False)
async def get_artists_view():
    return list(map(lambda x: x.name, mlflow_client.list_registered_models()))


@app.get('/', include_in_schema=False)
async def index_view(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/add-artist', include_in_schema=False)
async def add_artist_view(request: Request):
    return templates.TemplateResponse('train_index.html', {'request': request})


if __name__ == '__main__':
    uvicorn.run('application:app', host='0.0.0.0', port=3001, debug=True)
