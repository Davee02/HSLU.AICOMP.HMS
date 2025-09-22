import logging

from ..datasets.base_dataset import BaseDataset
from ..utils.base_factory import BaseFactory


class DatasetFactory(BaseFactory):
    """The factory class for creating various datasets."""

    base_class = BaseDataset
    registry = {}
    logger = logging.getLogger("DatasetFactory")
