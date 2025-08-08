import gc
import psutil
import sys
import torch
import logging

logger = logging.getLogger(__name__)   


def ensure_gpu_or_die():
    if not torch.cuda.is_available():
        logger.error(
            "No CUDA-capable GPU detected -- aborting run. "
            "Set up a GPU or remove this safety check."
        )
        sys.exit(1)          
    else:
        log_device_info()


def log_device_info():
    dev_id   = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(dev_id)
    logger.info("") 
    logger.info("Hardware configuration:")
    logger.info("======================")
    logger.info("Device: GPU %d – %s", dev_id, dev_name)
    logger.info("CUDA runtime: %s | PyTorch CUDA build: %s",
                torch.version.cuda, torch.version.git_version[:7])
    logger.info("GPUs visible: %d", torch.cuda.device_count())
    

def log_mem():
    logger.info(f"CPU Memory Used: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info(f"Total GPU memory: {total_mem / 1024**2:.2f} MB")
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def memory_cleanup(model, X_train, X_test, y_train, y_test):
    del model
    del X_train, X_test, y_train, y_test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_device() -> str:
    try:
        if torch.cuda.is_available():
            return f"cuda"
    except ModuleNotFoundError:
        pass
    return "cpu"


def count_gpus() -> int:
    return torch.cuda.device_count()