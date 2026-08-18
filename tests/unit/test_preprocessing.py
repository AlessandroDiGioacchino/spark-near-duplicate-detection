
import pytest

from spark_near_duplicates.preprocessing import normalize_text


@pytest.mark.parametrize(
    ( 'source', 'expected' ),
    [
        ( 'Great FOOD', 'great food' ),
        ( 'Hello, world!', 'hello world' ),
        ( 'Room 123 was clean', 'room was clean' ),
        ( '  too   much whitespace ', 'too much whitespace' ),
        ( '', '' ),
    ],
)
def test_default_normalization( source, expected ):
    assert normalize_text( source ) == expected

def test_handles_unicode():
    assert normalize_text( 'Caffè È BUONO!' ) == 'caffè è buono'

def test_rejects_null_text():
    with pytest.raises( TypeError ):
        normalize_text( None )
