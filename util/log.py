import logging
import sys
import os
from termcolor import colored

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name", "") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log
    
    
def setup_logging(output=None, distributed_rank=0, abbrev_name="PROB"):
    """初始化时传入distributed_rank, 只在rank 0 输出 INFO 及以上日志，其他 rank 输出 ERROR 及以上日志；所有 rank 都记录 DEBUG 及以上日志到文件"""
    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件处理器：所有进程记录 DEBUG 及以上
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)   
    
    # 控制台处理器：仅 rank 0 输出 INFO 及以上，其他输出 WARNING 及以上
    console = logging.StreamHandler(sys.stdout)
    if distributed_rank == 0:
        formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                abbrev_name=str(abbrev_name),
            )
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.ERROR)
    logger.addHandler(console)

