#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from multiprocessing.pool import ThreadPool

import base64
import logging

import requests

from . import settings
from .proto import *

logger = logging.getLogger(__name__)


def batch_download_images(image_links):
    def _download(image_link):
        attempts = settings.DEFAULT_DOWNLOAD_IMAGE_ATTEMPTS
        while attempts > 0:
            try:
                response = requests.get(
                    image_link, timeout=settings.DEFAULT_DOWNLOAD_ONE_IMAGE_TIMEOUT)
                return response.content
            except Exception:
                logger.warning("Failed to download image, retrying...")
            attempts -= 1
        return None

    pool = ThreadPool()
    applied = []
    for img_url in image_links:
        applied.append(pool.apply_async(_download, (img_url, )))
    ret = [r.get() for r in applied]
    return [_ for _ in ret if _ is not None]


def save_image(image_bytes, name):
    with open(name, "wb") as f:
        f.write(image_bytes)


def read_image(name):
    with open(name, "rb") as f:
        return f.read()


def read_image_to_base64(name):
    with open(name, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def add_lora_to_prompt(prompt: str, lora_name: str, weight: float = 1.0) -> str:
    prompt_split = [s.strip() for s in prompt.split(",")]
    ret = []
    replace = False
    for prompt_chunk in prompt_split:
        if prompt_chunk.startswith("<lora:{}".format(lora_name)):
            ret.append("<lora:{}:{}>".format(lora_name, weight))
            replace = True
        else:
            ret.append(prompt_chunk)
    if not replace:
        ret.append("<lora:{}:{}>".format(lora_name, weight))
    return ", ".join(ret)
