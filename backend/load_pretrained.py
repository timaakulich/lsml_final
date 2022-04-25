import sys
import glob
import pathlib
import torch

path = pathlib.Path(__file__).parent
sys.path.append(str(path / '..'))


from backend.nn_model import SimpleLSTM, mlflow, mlflow_client, production_stage  # noqa


def load():
    for model_path in glob.glob(str(path / 'models' / '*')):
        state, args, kwargs = torch.load(model_path)
        model = SimpleLSTM(*args, **kwargs)
        model.load_state_dict(state)
        artist = model_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        mlflow.pytorch.log_model(model, "models", registered_model_name=artist)
        last_model = max(mlflow_client.get_registered_model(artist).latest_versions, key=lambda x: int(x.version))
        mlflow_client.transition_model_version_stage(artist, last_model.version, production_stage)


if __name__ == '__main__':
    load()
