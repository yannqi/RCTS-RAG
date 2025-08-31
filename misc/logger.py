import os
from omegaconf import OmegaConf
import datetime
import logging
logger = logging.getLogger("RAG")
# logging.getLogger("httpx").setLevel(logging.WARNING)
def create_logging(logger, output_dir):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    file_handler = logging.FileHandler(os.path.join(output_dir, "logs/log.log"))
    logger.addHandler(file_handler)

def create_output_folders(project_name, output_dir, config, with_time=True):
    
    if with_time:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        out_dir = os.path.join(output_dir, f"{project_name}_{now}")
    else:
        out_dir = os.path.join(output_dir, f"{project_name}")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/outputs", exist_ok=True)
    os.makedirs(f"{out_dir}/logs", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir