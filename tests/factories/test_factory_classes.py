import unittest

from parameterized import parameterized_class

from src.datasets.dataset_factory import DatasetFactory
from src.utils.base_factory import BaseFactory


# fmt: off
@BaseFactory.register()
class DummyClass:
    ...


@DatasetFactory.register()
class BaseDatasetDummyClass:
    ...
# fmt: on


class FactoryBaseTest(unittest.TestCase):
    __test__ = False
    factory_class = None
    dummy_class = None
    dummy_class_name = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.factory = cls.factory_class()


@parameterized_class(
    [
        {
            "name": "BaseFactory",
            "factory_class": BaseFactory,
            "dummy_class": DummyClass,
            "dummy_class_name": "DummyClass",
            "__test__": True,
        },
        {
            "name": "DatasetFactory",
            "factory_class": DatasetFactory,
            "dummy_class": BaseDatasetDummyClass,
            "dummy_class_name": "BaseDatasetDummyClass",
            "__test__": True,
        },
    ],
)
class TestFactories(FactoryBaseTest):
    def test_register_and_create(self):
        assert len(self.factory.registry) >= 1
        dummy_object = self.factory.create(self.dummy_class_name)
        self.assertIsInstance(
            dummy_object,
            self.dummy_class,
        )

    def test_key_error(self):
        self.assertRaises(
            KeyError,
            self.factory.create,
            class_name="ClassNotInRegistry",
        )


if __name__ == "__main__":
    unittest.main()
