from celery import Celery
from backend.conf import settings
from backend.nn_model import train_nn_model, predict_lyrics

celery_app = Celery(
    'ml_project',
    backend=settings.redis_dsn,
    broker=settings.redis_dsn,
)
celery_app.conf.task_routes = {
    'backend.tasks.train_nn_model_task': {'queue': 'train'},
    'backend.tasks.generate_lyrics_task': {'queue': 'generate'},
}


@celery_app.task(bind=True)
def train_nn_model_task(self, artist, lyrics, epochs):
    return train_nn_model(artist, lyrics, epochs, self.request.id)


@celery_app.task()
def generate_lyrics_task(artist, start_text, length):
    return predict_lyrics(artist, start_text, length)


celery_app.autodiscover_tasks()