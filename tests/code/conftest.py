import pytest

from madewithml.data import CustomPreprocessor


@pytest.fixture
def dataset_loc():
    return "https://raw.githubusercontent.com/bansal1600/Fine-Tune-SciBERT/main/datasets/dataset.csv"


@pytest.fixture
def preprocessor():
    return CustomPreprocessor()
