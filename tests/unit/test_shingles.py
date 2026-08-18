
import pytest

from spark_near_duplicates.shingles import extract_shingles


def test_extracts_character_shingles():
    assert extract_shingles( 'abcd', size=3 ) == { 'abc', 'bcd' }

def test_removes_duplicate_shingles():
    assert extract_shingles( 'aaaa', size=2 ) == { 'aa' }

def test_shingle_size_equal_to_text_length():
    assert extract_shingles( 'abc', size=3 ) == { 'abc' }

def test_text_shorter_than_shingle_size():
    assert extract_shingles( 'ab', size=3) == set()

def test_empty_text():
    assert extract_shingles( '', size=3 ) == set()

@pytest.mark.parametrize( 'size', [ 0, -1 ] )
def test_rejects_invalid_shingle_size( size ):
    with pytest.raises( ValueError ):
        extract_shingles( 'abc', size=size )
