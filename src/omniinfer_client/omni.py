#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging

from time import sleep

from .version import __version__

from .exceptions import *
from .proto import *

import requests
from . import settings


logger = logging.getLogger(__name__)


class OmniClient:
    """OmniClient is the main entry point for interacting with the Omni API."""

    def __init__(self, api_key):
        self.base_url = "http://api.omniinfer.io/v2"
        self.api_key = api_key
        self.session = requests.Session()

        if not self.api_key:
            raise ValueError("OMNI_API_KEY environment variable not set")

        # eg: {"all": [proto.ModelInfo], "checkpoint": [proto.ModelInfo], "lora": [proto.ModelInfo]}
        self._model_list_cache = None

    def _get(self, api_path, params=None) -> dict:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Omni-Key': self.api_key,
            'User-Agent': "omniinfer-python-sdk/{}".format(__version__),
            'Accept-Encoding': 'gzip, deflate',
        }

        logger.debug(f"[GET] params: {params}")

        response = self.session.get(
            self.base_url + api_path,
            headers=headers,
            params=params,
            timeout=settings.DEFAULT_REQUEST_TIMEOUT,
        )

        logger.debug(f"[GET] response: {response.content}")
        if response.status_code != 200:
            logger.error(f"Request failed: {response}")
            raise OmniResponseError(
                f"Request failed with status {response.status_code}")

        return response.json()

    def _post(self, api_path, data) -> dict:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Omni-Key': self.api_key,
            'User-Agent': "omniinfer-python-sdk/{}".format(__version__),
            'Accept-Encoding': 'gzip, deflate',
        }

        logger.debug(f"[POST] data: {data}")

        response = self.session.post(
            self.base_url + api_path,
            headers=headers,
            json=data,
            timeout=settings.DEFAULT_REQUEST_TIMEOUT,
        )

        logger.debug(f"[POST] response: {response.content}")
        if response.status_code != 200:
            logger.error(f"Request failed: {response}")
            raise OmniResponseError(
                f"Request failed with status {response.status_code}")

        return response.json()

    def txt2img(self, request: Txt2ImgRequest) -> Txt2ImgResponse:
        """Asynchronously generate images from request

        Args:
            request (Txt2ImgRequest)

        Returns:
            Txt2ImgResponse
        """
        response = self._post('/txt2img', request.to_dict())

        return Txt2ImgResponse.from_dict(response)

    def progress(self, task_id: str) -> ProgressResponse:
        """Progress of a task

        Args:
            task_id (str)

        Returns:
            ProgressResponse
        """
        response = self._get('/progress', {
            'task_id': task_id,
        })

        return ProgressResponse.from_dict(response)

    def img2img(self, request: Img2ImgRequest) -> Img2ImgResponse:
        """Asynchronously generate images from request

        Args:
            request (Img2ImgRequest): _description_

        Returns:
            Img2ImgResponse: _description_
        """
        response = self._post('/img2img', request.to_dict())

        return Img2ImgResponse.from_dict(response)

    def wait_for_task(self, task_id, wait_for=300) -> ProgressResponse:
        """Wait for a task to complete

        Args:
            task_id (_type_): _description_
            wait_for (int, optional): _description_. Defaults to 300.

        Raises:
            OmniTimeoutError: _description_

        Returns:
            ProgressResponse: _description_
        """
        i = 0

        while i < wait_for:
            logger.info(f"Waiting for task {task_id} to complete")

            progress = self.progress(task_id)

            logger.info(
                f"Task {task_id} progress eta_relative: {progress.data.eta_relative}")

            if progress.data.status.finished():
                logger.info(f"Task {task_id} completed")
                return progress

            sleep(1)
            i += 1

        raise OmniTimeoutError(
            f"Task {task_id} failed to complete in {wait_for} seconds")

    def sync_txt2img(self, request: Txt2ImgRequest, download_images=True) -> ProgressResponse:
        """Synchronously generate images from request, optionally download images

        Args:
            request (Txt2ImgRequest): _description_
            download_images (bool, optional): _description_. Defaults to True.

        Returns:
            ProgressResponse: _description_
        """
        response = self.txt2img(request)

        if response.data is None:
            return OmniResponseError(f"Text to Image generation failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id)
        if download_images:
            res.download_images()
        return res

    def sync_img2img(self, request: Img2ImgRequest, download_images=True) -> ProgressResponse:
        """Syncronously generate images from request, optionally download images

        Args:
            request (Img2ImgRequest): _description_
            download_images (bool, optional): _description_. Defaults to True.

        Returns:
            ProgressResponse: _description_
        """
        response = self.img2img(request)

        if response.data is None:
            return OmniResponseError(f"Image to Image generation failed with response {response.msg}, code: {response.code}")

        res = self.wait_for_task(response.data.task_id)
        if download_images:
            res.download_images()
        return res

    def models(self, refresh=False) -> ModelList:
        """Get list of models

        Args:
            refresh (bool, optional): _description_. Defaults to False.

        Returns:
            ModelList: _description_
        """

        if (self._model_list_cache is None or len(self._model_list_cache) == 0) or refresh:
            res = self._get('/models')

            # TODO: fix this
            res_controlnet = self._get(
                '/models', params={'type': 'controlnet'})
            res_vae = self._get('/models', params={'type': 'vae'})

            tmp = []
            tmp.extend(MoodelsResponse.from_dict(res).data.models)
            tmp.extend(MoodelsResponse.from_dict(res_controlnet).data.models)
            tmp.extend(MoodelsResponse.from_dict(res_vae).data.models)

            # In future /models maybe return all models, so we need to filter out duplicates
            tmp_set = set()
            models = []
            for m in tmp:
                if m.sd_name not in tmp_set:
                    tmp_set.add(m.sd_name)
                    models.append(m)

            self._model_list_cache = ModelList(models)

        return self._model_list_cache
