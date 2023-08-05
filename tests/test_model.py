from omniinfer_client import *
import os


def test_model_api():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    models = client.models()
    assert all([m.civitai_nsfw is True for m in models.filter_by_nsfw(True)])
    assert all([m.civitai_nsfw is False for m in models.filter_by_nsfw(False)])

    assert len(models. \
        filter_by_type(ModelType.LORA). \
        filter_by_nsfw(False). \
        filter_by_civitai_tags('anime'). \
        sort_by_civitai_rating()) > 0

    assert len(models.filter_by_type(ModelType.CHECKPOINT)) > 0
    assert len(models.filter_by_type(ModelType.LORA)) > 0
    assert len(models.filter_by_type(ModelType.TEXT_INVERSION)) > 0
    assert len(models.filter_by_type(ModelType.CONTROLNET)) > 0
