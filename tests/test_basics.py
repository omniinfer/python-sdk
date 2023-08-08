from omniinfer_client import *
from omniinfer_client.utils import save_image, read_image_to_base64
import os
from PIL import Image
import io


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


def test_txt2img_upscale_2x():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    res = client.sync_txt2img(Txt2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
        enable_hr=True,
        hr_scale=2.0
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    img = Image.open(io.BytesIO(res.data.imgs_bytes[0]))
    assert img.size == (1024, 1024)


def test_txt2img_upscale_specify_size():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    res = client.sync_txt2img(Txt2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
        enable_hr=True,
        hr_resize_x=768,
        hr_resize_y=768
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    test_path = os.path.join(os.path.abspath(
        os.path.dirname(__name__)), "tests/data")
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    img = Image.open(io.BytesIO(res.data.imgs_bytes[0]))
    assert img.size == (768, 768)


def test_upscale_2x():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    res = client.sync_img2img(Img2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    image = base64.b64encode(res.data.imgs_bytes[0]).decode('utf-8')
    upscale_req = UpscaleRequest(
        image=image,
        resize_mode=UpscaleResizeMode.SCALE,
        upscaling_resize=2
    )
    upscale_res = client.sync_upscale(upscale_req)
    assert (upscale_res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(upscale_res.data.imgs_bytes) == 1)
    img = Image.open(io.BytesIO(upscale_res.data.imgs_bytes[0]))
    assert img.size == (1024, 1024)


def test_upscale_specify_size():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    res = client.sync_img2img(Img2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    image = base64.b64encode(res.data.imgs_bytes[0]).decode('utf-8')
    upscale_req = UpscaleRequest(
        image=image,
        resize_mode=UpscaleResizeMode.SIZE,
        upscaling_resize_h=768,
        upscaling_resize_w=768
    )
    upscale_res = client.sync_upscale(upscale_req)
    assert (upscale_res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(upscale_res.data.imgs_bytes) == 1)
    img = Image.open(io.BytesIO(upscale_res.data.imgs_bytes[0]))
    assert img.size == (768, 768)


def test_upscale_multiple_upscaler():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    res = client.sync_img2img(Img2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)

    image = base64.b64encode(res.data.imgs_bytes[0]).decode('utf-8')
    upscale_req = UpscaleRequest(
        image=image,
        upscaler_1='R-ESRGAN 4x+',
        resize_mode=UpscaleResizeMode.SIZE,
        upscaling_resize_h=768,
        upscaling_resize_w=768,
        upscaler_2='Nearest',
        gfpgan_visibility=1,
    )
    upscale_res = client.sync_upscale(upscale_req)
    assert (upscale_res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(upscale_res.data.imgs_bytes) == 1)
    img = Image.open(io.BytesIO(upscale_res.data.imgs_bytes[0]))
    assert img.size == (768, 768)


def test_txt2img_custom_headers():
    client = OmniClient(os.getenv('OMNI_API_KEY'))
    client.set_extra_headers({"User-Agent": "test-custom-user-agent"})

    res = client.sync_img2img(Img2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
    ))

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)


def test_txt2img_with_callback():
    client = OmniClient(os.getenv('OMNI_API_KEY'))

    def callback(res: ProgressResponse):
        assert isinstance(res.data.progress, float)

    res = client.sync_txt2img(Txt2ImgRequest(
        model_name='dreamshaper_8_93211.safetensors',
        prompt='a dog flying in the sky',
        width=512,
        height=512,
        batch_size=1,
        cfg_scale=7.5,
        sampler_name=Samplers.EULER_A,
    ), callback=callback)

    assert (res.data.status == ProgressResponseStatusCode.SUCCESSFUL)
    assert (len(res.data.imgs_bytes) == 1)
