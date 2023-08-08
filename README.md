# Omniinfer Python SDK

Thanks to the initial contribution of [@shanginn](https://github.com/shanginn), we have made the decision to create this SDK.

this SDK is based on the official [API documentation](https://docs.omniinfer.io/)

**join our discord server for help**

[![](https://dcbadge.vercel.app/api/server/nzqq8UScpx)](https://discord.gg/nzqq8UScpx) 

## Installation

```bash
pip install omniinfer-client
```

## Quick Start

**Get api key refer to [https://docs.omniinfer.io/get-started](https://docs.omniinfer.io/get-started/)**

```python
import os
from omniinfer_client import OmniClient, Txt2ImgRequest, Samplers, ModelType, save_image

client = OmniClient(os.getenv('OMNI_API_KEY'))

req = Txt2ImgRequest(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a dog flying in the sky',
    batch_size=1,
    cfg_scale=7.5,
    height=1024,
    width=1024,
    sampler_name=Samplers.EULER_A,
)
save_image(client.sync_txt2img(req).data.imgs_bytes[0], 'output.png')
```

## Examples

[txt2img_with_lora.py](./examples/txt2img_with_lora.py)

```python
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
```

### Model Search

[model_search.py](./examples/model_search.py)

```python
from omniinfer_client import OmniClient, ModelType

client = OmniClient(os.getenv('OMNI_API_KEY'))

# filter by model type
print("lora count", len(client.models().filter_by_type(ModelType.LORA)))
print("checkpoint count", len(client.models().filter_by_type(ModelType.CHECKPOINT)))
print("textinversion count", len(
    client.models().filter_by_type(ModelType.TEXT_INVERSION)))
print("vae count", len(client.models().filter_by_type(ModelType.VAE)))
print("controlnet count", len(client.models().filter_by_type(ModelType.CONTROLNET)))


# filter by civitai tags
client.models().filter_by_civi_tags('anime')

# filter by nsfw
client.models().filter_by_nsfw(False)  # or True

# sort by civitai download
client.models().sort_by_civitai_download()

# chain filters
client.models().\
    filter_by_type(ModelType.CHECKPOINT).\
    filter_by_nsfw(False).\
    filter_by_civitai_tags('anime')
```

### ControlNet QRCode

[controlnet_qrcode.py](./examples/controlnet_qrcode.py)

```python
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
```

### Txt2Img with Hires.Fix

[txt2img_with_hiresfix.py](./examples/txt2img_with_hiresfix.py)

```python
import os

from omniinfer_client import *

client = OmniClient(os.getenv('OMNI_API_KEY'))
req = Txt2ImgRequest(
    model_name='dreamshaper_8_93211.safetensors',
    prompt='a dog flying in the sky',
    width=512,
    height=512,
    batch_size=1,
    cfg_scale=7.5,
    sampler_name=Samplers.EULER_A,
    enable_hr=True,
    hr_scale=2.0
)

res = client.sync_txt2img(req)
if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)

save_image(res.data.imgs_bytes[0], "txt2img-hiresfix-1024.png")
```


## Testing

```
export OMNI_API_KEY=<YOUR_API_KEY>

python -m pytest
```