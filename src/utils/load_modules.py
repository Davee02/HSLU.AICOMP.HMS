import importlib
import os.path
from typing import Iterable


def load_modules(module_files: Iterable[str], module_path: str = "{}"):
    # loop over all module files
    for mod in module_files:
        # check if file is a real module and exists
        if (
            os.path.isfile(mod)
            and not mod.endswith("__init__.py")
            and not os.path.basename(mod).startswith(("_", "."))
        ):
            # extract module name from file name and import it
            mod_name = os.path.basename(os.path.splitext(mod)[0])
            # import the module
            importlib.import_module(name=module_path.format(mod_name))
