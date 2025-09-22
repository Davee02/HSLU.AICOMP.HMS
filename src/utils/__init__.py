import glob
import logging
import os
from pathlib import Path

from src.utils.utils import initialise_logging_config

from ..utils.load_modules import load_modules

initialise_logging_config()
logging.getLogger("root").info("Initialising Utils")


src_name = Path(__file__).resolve().parent.parent.stem
module_path = f"{src_name}.utils." + "{}"
load_modules(
    module_files=glob.glob(os.path.join(os.path.dirname(__file__), "*.py")),
    module_path=module_path,
)
