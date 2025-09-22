import random
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy
import torch
from parameterized import parameterized

from src.utils.utils import get_library_root, set_seeds, walk_and_collect


class TestUtils(unittest.TestCase):
    path = None
    file_ending = ".png"
    count = 10

    @classmethod
    def setUpClass(cls):
        cls.path = Path(tempfile.mkdtemp())
        cls.create_files(cls.path, cls.file_ending, count=cls.count)
        # Make empty directory
        tempfile.mkdtemp(dir=cls.path)
        # Make subdirectory
        files_subdir = Path(tempfile.mkdtemp(dir=cls.path))
        filepath = get_library_root() / "tests" / "testutils" / "file.txt"
        for i in range(cls.count):
            shutil.copy(filepath, files_subdir / f"sample_{i}.wav")
            shutil.copy(filepath, files_subdir / f"sample_{i}.mp3")
            shutil.copy(filepath, files_subdir / f"sample_{i}.flac")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path)

    @parameterized.expand(
        (
            (42,),
            (1337,),
        )
    )
    def test_set_seed(self, seed):
        set_seeds(seed)
        state = random.getstate()
        np_state = numpy.random.get_state()
        torch_state = torch.random.get_rng_state()
        random_uniform = random.uniform(0.0, 1.0)
        numpy_random_uniform = numpy.random.uniform(0.0, 1.0, 10)
        torch_random = torch.rand(1, 10)
        set_seeds(seed)
        self.assertEqual(state, random.getstate())
        numpy.testing.assert_equal(np_state, numpy.random.get_state())
        self.assertTrue(
            torch.equal(
                torch_state,
                torch.random.get_rng_state(),
            ),
        )
        self.assertEqual(random_uniform, random.uniform(0.0, 1.0))
        numpy.testing.assert_array_equal(
            numpy_random_uniform,
            numpy.random.uniform(0.0, 1.0, 10),
        )
        self.assertTrue(
            torch.equal(
                torch_random,
                torch.rand(1, 10),
            ),
        )

    @parameterized.expand(
        (
            (None,),
            (random.Random().getstate(),),
        )
    )
    def test_set_seed_wrong_input(self, input):
        self.assertRaises(ValueError, set_seeds, seed=input)

    def test_walk_and_collect(self):
        files = walk_and_collect(
            str(self.path), extensions=[self.file_ending, ".wav", ".mp3"]
        )
        self.assertEqual(3 * self.count, len(files))

    def test_walk_and_collect_empty_directory(self):
        path = Path(tempfile.mkdtemp())
        files = walk_and_collect(str(path), extensions=[".wav", ".mp3"])
        print(files)
        self.assertEqual(0, len(files))

    def test_walk_and_collect_invalid_path(self):
        path = []
        self.assertRaises(
            TypeError, walk_and_collect, path, extensions=[".wav", ".mp3"]
        )

    @staticmethod
    def create_files(path: Path, ending: str, count: int):
        for index in range(count):
            with open(path / ("FileName" + str(index) + ending), "w"):
                ...


if __name__ == "__main__":
    unittest.main()
