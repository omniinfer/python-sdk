#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from dataclasses import dataclass, field
from typing import Optional, List
from .serializer import JSONe
from enum import Enum
from .utils import batch_download_images


# --------------- ControlNet ---------------

class ControlNetMode(Enum):
    BALANCED = 0
    PROMPT_IMPORTANCE = 1
    CONTROLNET_IMPORTANCE = 2

    def __str__(self):
        return self.name


class ControlNetResizeMode(Enum):
    JUST_RESIZE = 0
    RESIZE_OR_CORP = 1
    RESIZE_AND_FILL = 2

    def __str__(self):
        return self.name


class ControlNetPreprocessor(Enum):
    NULL = 'none'
    CANNY = 'canny'
    DEPTH = 'depth'
    DEPTH_LERES = 'depth_leres'
    DEPTH_LERES_PLUS_PLUS = 'depth_leres++'
    HED = 'hed'
    HED_SAFE = 'hed_safe'
    MEDIAPIPE_FACE = 'mediapipe_face'
    MLSD = 'mlsd'
    NORMAL_MAP = 'normal_map'
    OPENPOSE = 'openpose'
    OPENPOSE_HAND = 'openpose_hand'
    OPENPOSE_FACE = 'openpose_face'
    OPENPOSE_FACEONLY = 'openpose_faceonly'
    OPENPOSE_FULL = 'openpose_full'
    CLIP_VISION = 'clip_vision'
    COLOR = 'color'
    PIDINET = 'pidinet'
    PIDINET_SAFE = 'pidinet_safe'
    PIDINET_SKETCH = 'pidinet_sketch'
    PIDINET_SCRIBBLE = 'pidinet_scribble'
    SCRIBBLE_XDOG = 'scribble_xdog'
    SCRIBBLE_HED = 'scribble_hed'
    SEGMENTATION = 'segmentation'
    THRESHOLD = 'threshold'
    DEPTH_ZOE = 'depth_zoe'
    NORMAL_BAE = 'normal_bae'
    ONEFORMER_COCO = 'oneformer_coco'
    ONEFORMER_ADE20K = 'oneformer_ade20k'
    LINEART = 'lineart'
    LINEART_COARSE = 'lineart_coarse'
    LINEART_ANIME = 'lineart_anime'
    LINEART_STANDARD = 'lineart_standard'
    SHUFFLE = 'shuffle'
    TILE_RESAMPLE = 'tile_resample'
    INVERT = 'invert'
    LINEART_ANIME_DENOISE = 'lineart_anime_denoise'
    REFERENCE_ONLY = 'reference_only'
    REFERENCE_ADAIN = 'reference_adain'
    REFERENCE_ADAIN_PLUS_ATTN = 'reference_adain+attn'
    INPAINT = 'inpaint'
    INPAINT_ONLY = 'inpaint_only'
    INPAINT_ONLY_PLUS_LAMA = 'inpaint_only+lama'
    TILE_COLORFIX = 'tile_colorfix'
    TILE_COLORFIX_PLUS_SHARP = 'tile_colorfix+sharp'

    def __str__(self):
        return self.name


@dataclass
class ControlnetUnit(JSONe):
    model: str
    weight: Optional[float] = 1
    module: Optional[ControlNetPreprocessor] = ControlNetPreprocessor.NULL
    input_image: Optional[str] = None
    control_mode: Optional[ControlNetMode] = ControlNetMode.BALANCED
    resize_mode: Optional[ControlNetResizeMode] = ControlNetResizeMode.RESIZE_OR_CORP
    mask: Optional[str] = None
    processor_res: Optional[int] = 512
    threshold_a: Optional[int] = 64
    threshold_b: Optional[int] = 64
    guidance_start: Optional[float] = 0.0
    guidance_end: Optional[float] = 1.0
    pixel_perfect: Optional[bool] = False


# --------------- Samplers ---------------
@dataclass
class Samplers:
    EULER_A = 'Euler a'
    EULER = 'Euler'
    LMS = 'LMS'
    HEUN = 'Heun'
    DPM2 = 'DPM2'
    DPM2_A = 'DPM2 a'
    DPM2_KARRAS = 'DPM2 Karras'
    DPM2_A_KARRAS = 'DPM2 a Karras'
    DPMPP_S_A = 'DPM++ 2S a'
    DPMPP_M = 'DPM++ 2M'
    DPMPP_SDE = 'DPM++ SDE'
    DPMPP_KARRAS = 'DPM++ Karras'
    DPMPP_S_A_KARRAS = 'DPM++ 2S a Karras'
    DPMPP_M_KARRAS = 'DPM++ 2M Karras'
    DPMPP_SDE_KARRAS = 'DPM++ SDE Karras'
    DDIM = 'DDIM'
    PLMS = 'PLMS'
    UNIPC = 'UniPC'


# --------------- Text2Image ---------------


@dataclass
class Txt2ImgRequest(JSONe):
    prompt: str
    negative_prompt: Optional[str] = None
    model_name: str = 'dreamshaper_5BakedVae.safetensors'
    sampler_name: str = None
    batch_size: int = 1
    n_iter: int = 1
    steps: int = 20
    cfg_scale: float = 7
    height: int = 512
    width: int = 512
    seed: Optional[int] = -1
    restore_faces: Optional[bool] = False
    sd_vae: Optional[str] = None
    clip_skip: Optional[int] = 1
    controlnet_units: Optional[List[ControlnetUnit]] = None

    enable_hr: Optional[bool] = False
    hr_upscaler: Optional[str] = 'R-ESRGAN 4x+'
    hr_scale: Optional[float] = 2.0
    hr_resize_x: Optional[int] = None
    hr_resize_y: Optional[int] = None


class Txt2ImgResponseCode(Enum):
    NORMAL = 0
    INTERNAL_ERROR = -1
    INVALID_JSON = 1
    MODEL_NOT_EXISTS = 2
    TASK_ID_NOT_EXISTS = 3
    INVALID_AUTH = 4
    HOST_UNAVAILABLE = 5
    PARAM_RANGE_ERROR = 6
    COST_BALANCE_ERROR = 7
    SAMPLER_NOT_EXISTS = 8
    TIMEOUT = 9

    UNKNOWN = 100

    @classmethod
    def _missing_(cls, number):
        return cls(cls.UNKNOWN)


@dataclass
class Txt2ImgResponseData(JSONe):
    task_id: str
    warn: Optional[str] = None


@dataclass
class Txt2ImgResponse(JSONe):
    code: Txt2ImgResponseCode
    msg: str
    data: Optional[Txt2ImgResponseData] = None


#  --------------- Image2Image ---------------


@dataclass
class Img2ImgRequest(JSONe):
    model_name: str = 'dreamshaper_5BakedVae.safetensors'
    sampler_name: str = None
    init_images: List[str] = None
    mask: Optional[str] = None
    resize_mode: Optional[int] = 0
    denoising_strength: Optional[float] = 0.75
    cfg_scale: Optional[float] = None
    mask_blur: Optional[int] = 4
    inpainting_fill: Optional[int] = 1
    inpaint_full_res: Optional[int] = 0
    inpaint_full_res_padding: Optional[int] = 32
    inpainting_mask_invert: Optional[int] = 0
    initial_noise_multiplier: Optional[float] = 1.0
    prompt: Optional[str] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    batch_size: Optional[int] = 1
    n_iter: Optional[int] = 1
    steps: Optional[int] = 20
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    restore_faces: Optional[bool] = False
    sd_vae: Optional[str] = None
    clip_skip: Optional[int] = 1
    controlnet_units: Optional[List[ControlnetUnit]] = None


class Img2ImgResponseCode(Enum):
    NORMAL = 0
    INTERNAL_ERROR = -1
    INVALID_JSON = 1
    MODEL_NOT_EXISTS = 2
    TASK_ID_NOT_EXISTS = 3
    INVALID_AUTH = 4
    HOST_UNAVAILABLE = 5
    PARAM_RANGE_ERROR = 6
    COST_BALANCE_ERROR = 7
    SAMPLER_NOT_EXISTS = 8
    TIMEOUT = 9

    UNKNOWN = 100

    @classmethod
    def _missing_(cls, number):
        return cls(cls.UNKNOWN)


@dataclass
class Img2ImgResponseData(JSONe):
    task_id: str
    warn: Optional[str] = None


@dataclass
class Img2ImgResponse(JSONe):
    code: Img2ImgResponseCode
    msg: str
    data: Optional[Img2ImgResponseData] = None

# --------------- Progress ---------------


class ProgressResponseStatusCode(Enum):
    INITIALIZING = 0
    RUNNING = 1
    SUCCESSFUL = 2
    FAILED = 3
    TIMEOUT = 4

    UNKNOWN = 100

    @classmethod
    def _missing_(cls, number):
        return cls(cls.UNKNOWN)

    def finished(self):
        return self in (ProgressResponseStatusCode.SUCCESSFUL, ProgressResponseStatusCode.FAILED, ProgressResponseStatusCode.TIMEOUT)


@dataclass
class ProgressData(JSONe):
    status: ProgressResponseStatusCode
    progress: int
    eta_relative: int
    imgs: Optional[List[str]] = None
    imgs_bytes: Optional[List[str]] = None
    # info: Optional[Dict[str, str]] = None   # We'll handle this field separately
    failed_reason: Optional[str] = ""
    current_images: Optional[List[str]] = None
    submit_time: Optional[str] = ""
    execution_time: Optional[str] = ""
    txt2img_time: Optional[str] = ""
    finish_time: Optional[str] = ""


class ProgressResponseCode(Enum):
    NORMAL = 0
    INTERNAL_ERROR = -1
    INVALID_JSON = 1
    MODEL_NOT_EXISTS = 2
    TASK_ID_NOT_EXISTS = 3
    INVALID_AUTH = 4
    HOST_UNAVAILABLE = 5
    PARAM_RANGE_ERROR = 6
    COST_BALANCE_ERROR = 7
    SAMPLER_NOT_EXISTS = 8
    TIMEOUT = 9

    UNKNOWN = 100

    @classmethod
    def _missing_(cls, number):
        return cls(cls.UNKNOWN)


@dataclass
class ProgressResponse(JSONe):
    code: ProgressResponseCode
    data: Optional[ProgressData] = None
    msg: Optional[str] = ""

    def download_images(self):
        if self.data.imgs is not None and len(self.data.imgs) > 0:
            self.data.imgs_bytes = batch_download_images(self.data.imgs)

# --------------- Upscale ---------------


class UpscaleResizeMode(Enum):
    SCALE = 0
    SIZE = 1


@dataclass
class UpscaleRequest(JSONe):
    image: str
    upscaler_1: Optional[str] = 'R-ESRGAN 4x+'
    resize_mode: Optional[UpscaleResizeMode] = UpscaleResizeMode.SCALE
    upscaling_resize: Optional[float] = 2.0
    upscaling_resize_w: Optional[int] = None
    upscaling_resize_h: Optional[int] = None
    upscaling_crop: Optional[bool] = False

    upscaler_2: Optional[str] = None
    extras_upscaler_2_visibility: Optional[float] = None
    gfpgan_visibility: Optional[float] = None
    codeformer_visibility: Optional[float] = None
    codeformer_weight: Optional[float] = None


class UpscaleResponseCode(Enum):
    NORMAL = 0
    INTERNAL_ERROR = -1
    INVALID_JSON = 1
    MODEL_NOT_EXISTS = 2
    TASK_ID_NOT_EXISTS = 3
    INVALID_AUTH = 4
    HOST_UNAVAILABLE = 5
    PARAM_RANGE_ERROR = 6
    COST_BALANCE_ERROR = 7
    SAMPLER_NOT_EXISTS = 8
    TIMEOUT = 9

    UNKNOWN = 100

    @classmethod
    def _missing_(cls, number):
        return cls(cls.UNKNOWN)


@dataclass
class UpscaleResponseData(JSONe):
    task_id: str
    warn: Optional[str] = None


@dataclass
class UpscaleResponse(JSONe):
    code: UpscaleResponseCode
    msg: str
    data: Optional[UpscaleResponseData] = None


# --------------- Model ---------------


class ModelType(Enum):
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    CONTROLNET = "controlnet"
    TEXT_INVERSION = "textualinversion"
    UPSCALER = "upscaler"

    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, number):
        return cls(cls.UNKNOWN)


@dataclass
class CivitaiImageMeta(JSONe):
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    sampler_name: Optional[str] = None
    steps: Optional[int] = None
    cfg_scale: Optional[int] = None
    seed: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    model_name: Optional[str] = None


@dataclass
class CivitaiImage(JSONe):
    url: str
    nsfw: str
    meta: Optional[CivitaiImageMeta] = None


@dataclass
class ModelInfo(JSONe):
    name: str
    hash: str
    civitai_version_id: int
    sd_name: str
    third_source: str
    download_status: int
    download_name: str
    dependency_status: int
    type: ModelType
    civitai_nsfw: Optional[bool] = False
    civitai_model_id: Optional[int] = 0
    civitai_link: Optional[str] = None
    civitai_images: Optional[List[CivitaiImage]] = field(default_factory=lambda: [])
    civitai_download_url: Optional[str] = None
    civitai_allow_commercial_use: Optional[bool] = True
    civitai_allow_different_license: Optional[bool] = True
    civitai_create_at: Optional[str] = None
    civitai_update_at: Optional[str] = None
    civitai_tags: Optional[str] = None
    civitai_download_count: Optional[int] = 0
    civitai_favorite_count: Optional[int] = 0
    civitai_comment_count: Optional[int] = 0
    civitai_rating_count: Optional[int] = 0
    civitai_rating: Optional[float] = 0.0
    omni_used_count: Optional[int] = None
    civitai_image_url:  Optional[str] = None
    civitai_image_nsfw:  Optional[bool] = False
    civitai_origin_image_url:  Optional[str] = None
    civitai_image_prompt:  Optional[str] = None
    civitai_image_negative_prompt:  Optional[str] = None
    civitai_image_sampler_name:  Optional[str] = None
    civitai_image_height: Optional[int] = None
    civitai_image_width:  Optional[int] = None
    civitai_image_steps:  Optional[int] = None
    civitai_image_cfg_scale:  Optional[int] = None
    civitai_image_seed:  Optional[int] = None


@dataclass
class ModelData(JSONe):
    models: List[ModelInfo] = None


@dataclass
class MoodelsResponse(JSONe):
    code: int
    msg: str
    data: Optional[ModelData] = field(default_factory=lambda: [])


class ModelList(list):
    """A list of ModelInfo"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_by_civitai_version_id(self, civitai_version_id: int):
        for model in self:
            if model.civitai_version_id == civitai_version_id:
                return model
        return None

    def get_by_name(self, name):
        for model in self:
            if model.name == name:
                return model
        return None

    def get_by_sd_name(self, sd_name):
        for model in self:
            if model.sd_name == sd_name:
                return model
        return None

    def list_civitai_tags(self) -> List[str]:
        s = set()
        for model in self:
            if model.civitai_tags:
                s.update(s.strip()
                         for s in model.civitai_tags.split(",") if s.strip())
        return list(s)

    def filter_by_civitai_tags(self, *tags):
        ret = []
        for model in self:
            if model.civitai_tags:
                if set(tags).issubset(set(s.strip() for s in model.civitai_tags.split(","))):
                    ret.append(model)
        return ModelList(ret)

    def filter_by_nsfw(self, nsfw: bool):
        return ModelList([model for model in self if model.civitai_nsfw == nsfw])

    def filter_by_type(self, type):
        return ModelList([model for model in self if model.type == type])

    def filter_by_civitai_model_id(self, civitai_model_id: int):
        return ModelList([model for model in self if model.civitai_model_id == civitai_model_id])

    def filter_by_civitai_model_name(self, name: str):
        return ModelList([model for model in self if model.name == name])

    def sort_by_civitai_download(self):
        return ModelList(sorted(self, key=lambda x: x.civitai_download_count, reverse=True))

    def sort_by_civitai_rating(self):
        return ModelList(sorted(self, key=lambda x: x.civitai_rating, reverse=True))

    def sort_by_civitai_favorite(self):
        return ModelList(sorted(self, key=lambda x: x.civitai_favorite, reverse=True))

    def sort_by_civitai_comment(self):
        return ModelList(sorted(self, key=lambda x: x.civitai_comment, reverse=True))
