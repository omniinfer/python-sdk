#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from dataclass_wizard import JSONWizard, DumpMeta


class JSONe(JSONWizard):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DumpMeta(key_transform='SNAKE').bind_to(cls)
