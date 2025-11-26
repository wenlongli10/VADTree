import copy
import os,sys
import argparse
import numpy as np
import json
from pathlib import Path

def args_to_dict(args, exclude_keys=None, max_depth=3):
    """
    将args对象转换为JSON可序列化的字典
    :param args: 参数对象（Namespace/class/dict）
    :param exclude_keys: 需要排除的键列表
    :param max_depth: 最大递归深度防止循环引用
    :return: 可序列化的字典
    """
    def convert_value(value, current_depth):
        if current_depth >= max_depth:
            return str(value)  # 防止无限递归

        # 处理常见不可序列化类型
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, (list, tuple, set)):
            return [convert_value(v, current_depth+1) for v in value]
        if isinstance(value, dict):
            return {k: convert_value(v, current_depth+1) for k, v in value.items()}

        # 处理对象属性
        try:
            return {
                k: convert_value(v, current_depth+1)
                for k, v in vars(value).items()
                if not k.startswith('_')  # 排除私有属性
            }
        except TypeError:
            return str(value)

    # 获取参数字典
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)  # 适用于argparse.Namespace或普通对象

    # 过滤排除项
    exclude = set(exclude_keys or [])
    filtered = {k: v for k, v in args_dict.items() if k not in exclude}

    # 递归转换值
    return convert_value(filtered, current_depth=0)
