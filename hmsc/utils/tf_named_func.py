#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:34:09 2023

@author: gtikhono
"""

import tensorflow as tf

def tf_named_func(name):
    def decorator(original_func):
        def decorated_func(*args, **kwargs):
            with tf.name_scope(name) as scope:
                result = original_func(*args, **kwargs)
            return result
        return decorated_func
    
    return decorator        