import pytest
from sota_rome.tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer()

def test_tokenizer(tokenizer):
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
