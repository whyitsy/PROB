import logging
import sys
import os



def setup_logging(output=None, distributed_rank=0):
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
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器：所有进程记录 DEBUG 及以上
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)   
    
    # 控制台处理器：仅 rank 0 输出 INFO 及以上，其他输出 WARNING 及以上
    console = logging.StreamHandler(sys.stdout)
    if distributed_rank == 0:
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.ERROR)
    console_formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')  # 可简化
    console.setFormatter(console_formatter)
    logger.addHandler(console)
