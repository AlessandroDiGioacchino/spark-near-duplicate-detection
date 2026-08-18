
import pytest

from spark_near_duplicates.minhash import minhash_signature
from spark_near_duplicates.similarity import estimate_similarity


def test_signature_has_requested_length():
  signature = minhash_signature(
    { 'abc', 'bcd' },
    num_hashes=16,
    seed=42,
  )

  assert len( signature ) == 16
  assert all( isinstance( value, int ) for value in signature )

def test_signature_is_deterministic():
  arguments = {
    'shingles': { 'abc', 'bcd' },
    'num_hashes': 16,
    'seed': 42,
  }

  assert minhash_signature( **arguments ) == minhash_signature( **arguments )

def test_set_iteration_order_does_not_affect_signature():
  left = minhash_signature(
    { 'abc', 'bcd', 'cde' },
    num_hashes=16,
    seed=42,
  )

  right = minhash_signature(
    set( reversed([ 'abc', 'bcd', 'cde' ]) ),
    num_hashes=16,
    seed=42,
  )

  assert left == right

def test_identical_shingles_have_identical_signatures():
  left = minhash_signature( { 'abc', 'bcd' }, num_hashes=32, seed=42 )
  right = minhash_signature( { 'abc', 'bcd' }, num_hashes=32, seed=42 )

  assert left == right

def test_different_seeds_change_signature():
    first = minhash_signature( { 'abc', 'bcd' }, num_hashes=16, seed=1 )
    second = minhash_signature( { 'abc', 'bcd' }, num_hashes=16, seed=2 )

    assert first != second

@pytest.mark.parametrize( 'num_hashes', [ 0, -1 ] )
def test_rejects_nonpositive_hash_count( num_hashes ):
  with pytest.raises( ValueError ):
    minhash_signature(
      { 'abc' },
      num_hashes=num_hashes,
      seed=42,
    )

def test_rejects_empty_shingle_set():
  with pytest.raises( ValueError ):
    minhash_signature(
        set(),
        num_hashes=16,
        seed=42,
    )

def test_rejects_non_string_shingles():
  with pytest.raises(TypeError ):
    minhash_signature(
      { 'abc', 123 },
      num_hashes=16,
      seed=42,
    )

def test_minhash_estimate_approximates_jaccard():
    left = {f"item-{i}" for i in range(100)}
    right = {f"item-{i}" for i in range(50, 150)}

    left_signature = minhash_signature(left, num_hashes=512, seed=42)
    right_signature = minhash_signature(right, num_hashes=512, seed=42)

    estimate = estimate_similarity( left_signature, right_signature )

    # Exact Jaccard is 50 / 150 == 1/3.
    assert estimate == pytest.approx(1 / 3, abs=0.08)
