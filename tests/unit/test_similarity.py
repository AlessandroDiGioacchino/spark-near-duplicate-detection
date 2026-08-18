
import pytest

from spark_near_duplicates.similarity import (
  estimate_similarity,
  jaccard_similarity
)


def test_identical_sets_have_similarity_one():
    assert jaccard_similarity({ 'ab', 'bc' }, { 'ab', 'bc' }) == 1.0

def test_disjoint_sets_have_similarity_zero():
    assert jaccard_similarity({ 'ab' }, { 'cd' }) == 0.0

def test_partial_overlap():
    result = jaccard_similarity({ 'a', 'b' }, { 'b', 'c' })
    assert result == pytest.approx( 1/3 )

def test_similarity_is_symmetric():
    left = { 'a', 'b' }
    right = { 'b', 'c', 'd' }

    assert ( jaccard_similarity( left, right ) ==
             jaccard_similarity( right, left ) )

def test_two_empty_sets_are_identical():
    assert jaccard_similarity( set(), set() ) == 1.0


def test_estimate_compares_corresponding_positions():
    left = ( 1, 2, 3, 4 )
    right = ( 1, 9, 3, 8 )

    assert estimate_similarity( left, right ) == 0.5

def test_identical_signatures_have_similarity_one():
    signature = ( 10, 20, 30 )
    assert estimate_similarity( signature, signature ) == 1.0

def test_rejects_different_signature_lengths():
    with pytest.raises( ValueError ):
        estimate_similarity(( 1, 2 ), ( 1, 2, 3 ))

def test_rejects_empty_signatures():
    with pytest.raises( ValueError ):
        estimate_similarity((), ())
