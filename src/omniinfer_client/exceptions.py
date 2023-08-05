#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class OmniError(Exception):
    pass


class OmniResponseError(OmniError):
    pass


class OmniTimeoutError(OmniError):
    pass
