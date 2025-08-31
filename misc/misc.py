import os
import base64
from hashlib import md5
import asyncio
from misc import logger
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()



def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """获取当前运行的事件循环，如果没有运行的循环，就创建一个新的事件循环并设置。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop



