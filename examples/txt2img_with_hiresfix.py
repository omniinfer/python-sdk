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
