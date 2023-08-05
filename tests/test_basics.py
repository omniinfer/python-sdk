from omniinfer_client import *
from omniinfer_client.utils import save_image, read_image_to_base64
import os


import pytest


@pytest.mark.dependency()
def test_txt2img_sync():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    res = client.sync_txt2img(Txt2ImgRequest(
        prompt='a dog flying in the sky',
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    save_image(res.data.imgs_bytes[0], os.path.join(
        test_path, 'test_txt2img_sync.png'))


@pytest.mark.dependency(depends=['test_txt2img_sync'])
def test_img2img_sync():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    init_image = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data/test_txt2img_sync.png")
    init_image_base64 = read_image_to_base64(init_image)

    res = client.sync_img2img(Img2ImgRequest(
        prompt='a dog flying in the sky',
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
        init_images=[init_image_base64]
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    save_image(res.data.imgs_bytes[0], os.path.join(
        test_path, 'test_img2img_sync.png'))


@pytest.mark.dependency(depends=['test_img2img_sync'])
def test_txt2img_controlnet():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    init_image = os.path.join(os.path.abspath(os.path.dirname(
        __name__)), "tests/data/test_txt2img_sync.png")
    init_image_base64 = read_image_to_base64(init_image)

    client = OmniClient(os.getenv('OMNI_API_KEY'))
    request = Txt2ImgRequest(
        prompt='a dog flying in the sky',
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
        controlnet_units=[
            ControlnetUnit(
                input_image=init_image_base64,
                model='control_v11p_sd15_canny',
                module='canny',
            ),
        ]
    )

    res = client.sync_txt2img(request)
    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    assert len(res.data.imgs_bytes) == 2

    save_image(res.data.imgs_bytes[1], os.path.join(
        test_path, 'test_txt2img_controlnet_processor.png'))
    save_image(res.data.imgs_bytes[0], os.path.join(
        test_path, 'test_txt2img_controlnet_result.png'))


@pytest.mark.dependency(depends=['test_img2img_sync'])
def test_img2img_controlnet():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    init_image = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data/test_txt2img_sync.png")
    init_image_base64 = read_image_to_base64(init_image)

    res = client.sync_img2img(Img2ImgRequest(
        prompt='a dog flying in the sky',
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
        init_images=[init_image_base64],
        controlnet_units=[
            ControlnetUnit(
                input_image=init_image_base64,
                model='control_v11p_sd15_canny',
                module='canny',
            ),
        ]
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert len(res.data.imgs_bytes) == 2

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    save_image(res.data.imgs_bytes[1], os.path.join(
        test_path, 'test_img2img_controlnet_processor.png'))
    save_image(res.data.imgs_bytes[0], os.path.join(
        test_path, 'test_img2img_controlnet_result.png'))
