
import pytest

from spark_near_duplicates.lsh import (
  band_signature,
  generate_candidate_pairs
)


def test_emits_one_record_per_band():
  records = band_signature(
    'document-a',
    ( 10, 11, 20, 21, 30, 31 ),
    num_bands=3
  )

  assert len( records ) == 3
  assert [ key[ 0 ] for key, _ in records ] == [ 0, 1, 2 ]
  assert all( document_id == 'document-a' for _, document_id in records )

def test_identical_signatures_share_all_bucket_keys():
    signature = ( 10, 11, 20, 21, 30, 31 )

    left_records = band_signature(
        'document-a',
        signature,
        num_bands=3
    )

    right_records = band_signature(
        'document-b',
        signature,
        num_bands=3
    )

    left_keys = [ key for key, _ in left_records ]
    right_keys = [ key for key, _ in right_records ]

    assert left_keys == right_keys

def test_changing_one_band_only_changes_its_bucket():
  left_records = band_signature(
    'document-a',
    ( 10, 11, 20, 21, 30, 31 ),
    num_bands=3
  )

  right_records = band_signature(
    'document-b',
    ( 99, 11, 20, 21, 30, 31 ),
    num_bands=3
  )

  left_keys = [ key for key, _ in left_records ]
  right_keys = [ key for key, _ in right_records ]

  assert left_keys[ 0 ] != right_keys[ 0 ]
  assert left_keys[ 1 ] == right_keys[ 1 ]
  assert left_keys[ 2 ] == right_keys[ 2 ]

def test_rejects_uneven_bands():
  with pytest.raises( ValueError ):
    band_signature(
      'document-a',
      tuple( range( 10 ) ),
      num_bands=3
    )

@pytest.mark.parametrize( 'num_bands', [ 0, -1 ] )
def test_rejects_nonpositive_band_count( num_bands ):
  with pytest.raises( ValueError ):
    band_signature(
      'document-a',
      ( 10, 11 ),
      num_bands=num_bands
    )

def test_rejects_empty_signature():
  with pytest.raises( ValueError ):
    band_signature(
      'document-a',
      (),
      num_bands=2
    )


def test_generates_pair_from_shared_bucket():
  records = [
    (( 0, 'bucket-a' ), 'document-1' ),
    (( 0, 'bucket-a' ), 'document-2' )
  ]

  assert generate_candidate_pairs( records ) == {
    ( 'document-1', 'document-2' )
  }

def test_does_not_pair_documents_in_different_buckets():
  records = [
    (( 0, 'bucket-a' ), 'document-1' ),
    (( 0, 'bucket-b' ), 'document-2' )
  ]

  assert generate_candidate_pairs( records ) == set()

def test_deduplicates_pair_from_multiple_band_collisions():
  records = [
    (( 0, 'bucket-a'), 'document-1' ),
    (( 0, 'bucket-a'), 'document-2' ),
    (( 1, 'bucket-b'), 'document-1' ),
    (( 1, 'bucket-b'), 'document-2' )
  ]

  assert generate_candidate_pairs( records ) == {
    ( 'document-1', 'document-2' )
  }
