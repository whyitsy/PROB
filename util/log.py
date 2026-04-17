import logging
import sys
import os
from termcolor import colored

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name", "root")
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = f"{record.name}({self._abbrev_name})" if self._abbrev_name else record.name
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log
    
    
def setup_logging(output=None, distributed_rank=0, abbrev_name="PROB"):
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
    
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s]-%(name)s-%(levelname)s-%(message)s',datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)   
    
    console = logging.StreamHandler(sys.stdout)
    if distributed_rank == 0:
        formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                abbrev_name=str(abbrev_name),
            )
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.ERROR)
    logger.addHandler(console)

