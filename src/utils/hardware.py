import gc
import os
import multiprocessing  
import psutil
import sys
import torch
import logging

logger = logging.getLogger(__name__)   


def check_resource_availability(req_device, num_req_cpus, num_req_gpus):
    """Aborts run if not enough resources available"""
    num_req_gpus = num_req_gpus or 0
    num_req_cpus = num_req_cpus or 0

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else: # fallback for non-slurm runs
        num_avail_cpus = multiprocessing.cpu_count()
    num_avail_gpus = torch.cuda.device_count()

    abort = False
    if num_req_gpus > 0 or req_device=="cuda":
        if num_avail_gpus < num_req_gpus:
            logger.error(f"Number of available GPUs ({num_avail_gpus}) is smaller than requested {num_req_gpus} aborting run.")
            abort=True
    if num_req_cpus > 0 or req_device=="cpu":
        if num_avail_cpus < num_req_cpus:
            logger.error(f"Number of available CPUs ({num_avail_cpus}) is smaller than requested {num_req_cpus} aborting run.")
            abort=True
    if abort:
        sys.exit(1)
    return num_avail_cpus, num_avail_gpus               


def set_hardware_config(config):
    num_req_cpus = config["num_cpus"]
    num_req_gpus = config["num_gpus"]
    req_device = config["device"]

    num_avail_cpus, num_avail_gpus = check_resource_availability(req_device=req_device,
                                num_req_cpus=num_req_cpus,
                                num_req_gpus=num_req_gpus
    )
    
    if num_req_gpus is None: # number of gpus unspecified
        if req_device == "cuda":
            config["num_gpus"] = num_avail_gpus
        else:
            config["num_gpus"] = 0
    if num_req_cpus is None: # number of cpus unspecified
        config["num_cpus"] = num_avail_cpus
    
    log_device_info(req_device, config["num_gpus"], config["num_cpus"])


def log_device_info(req_device, num_req_gpus, num_req_cpus):
    logger.info("") 
    logger.info("Hardware configuration:")
    logger.info("======================")
    if req_device == "cuda":
        dev_id   = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_id)
        logger.info("Current device: GPU %d – %s", dev_id, dev_name)
        logger.info("CUDA runtime: %s | PyTorch CUDA build: %s",
                    torch.version.cuda, torch.version.git_version[:7])
        logger.info(f"Requested GPUs in use: {num_req_gpus}")
    # else:
        # dev_id   = torch.cpu.current_device()
        # dev_name = torch.cpu.get_device_name(dev_id)
        # logger.info("Current device: CPU %d – %s", dev_id, dev_name)
    logger.info(f"Requested CPUs in use: {num_req_cpus}")


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

def count_cpus() -> int:
    return torch.cpu.device_count()