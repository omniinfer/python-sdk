import os

from omniinfer_client import *

client = OmniClient(os.getenv('OMNI_API_KEY'))
req = Txt2ImgRequest(
    model_name='sd_xl_base_1.0.safetensors',
    prompt='a dog flying in the sky',
    width=1024,
    height=1024,
    batch_size=1,
    cfg_scale=7.5,
    sampler_name=Samplers.EULER_A,
    sd_refiner=Refiner(
        checkpoint='sd_xl_refiner_1.0.safetensors',
        switch_at=0.5,
    ))

res = client.sync_txt2img(req)
if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
    raise Exception('Failed to generate image with error: ' +
                    res.data.failed_reason)

save_image(res.data.imgs_bytes[0], "txt2img-refiner.png")
