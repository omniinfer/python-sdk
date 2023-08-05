#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

from omniinfer_client import *

# get your api key refer to https://docs.omniinfer.io/get-started/
client = OmniClient(os.getenv('OMNI_API_KEY'))

controlnet_model = client.models().filter_by_type(ModelType.CONTROLNET).get_by_name("control_v1p_sd15_qrcode_monster_v2")
if controlnet_model is None:
    raise Exception("controlnet model not found")

req = Txt2ImgRequest(
    prompt="a beautify butterfly in the colorful flowers, best quality, best details, masterpiece",
    sampler_name=Samplers.DPMPP_M_KARRAS,
    width=512,
    height=512,
    steps=30,
    controlnet_units=[
        ControlnetUnit(
            input_image=read_image_to_base64(os.path.join(os.path.abspath(os.path.dirname(__file__)), "fixtures/qrcode.png")),
            control_mode=ControlNetMode.BALANCED,
            model=controlnet_model.sd_name,
            module=ControlNetPreprocessor.NULL,
            resize_mode=ControlNetResizeMode.JUST_RESIZE,
            weight=2.0,
        )
    ]
)

res = client.sync_txt2img(req)
if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)

save_image(res.data.imgs_bytes[0], "qrcode-art.png")
