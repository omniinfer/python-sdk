#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from omniinfer_client import OmniClient, Txt2ImgRequest, Samplers, ProgressResponseStatusCode, ModelType, add_lora_to_prompt, save_image


client = OmniClient(os.getenv('OMNI_API_KEY'))
models = client.models()

# Anything V5/Ink, https://civitai.com/models/9409/or-anything-v5ink
checkpoint_model = models.filter_by_type(ModelType.CHECKPOINT).get_by_civitai_version_id(90854)

# Detail Tweaker LoRA, https://civitai.com/models/58390/detail-tweaker-lora-lora
lora_model = models.filter_by_type(ModelType.LORA).get_by_civitai_version_id(62833)

prompt = add_lora_to_prompt('a dog flying in the sky', lora_model.sd_name, "0.8")

res = client.sync_txt2img(Txt2ImgRequest(
    prompt=prompt,
    batch_size=1,
    cfg_scale=7.5,
    sampler_name=Samplers.EULER_A,
    model_name=checkpoint_model.sd_name,
    seed=103304,
))

if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)
save_image(res.data.imgs_bytes[0], "test.png")
