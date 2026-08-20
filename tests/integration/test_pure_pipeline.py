
from spark_near_duplicates.lsh import (
    band_signature,
    generate_candidate_pairs,
)

from spark_near_duplicates.minhash import minhash_signature
from spark_near_duplicates.preprocessing import normalize_text
from spark_near_duplicates.shingles import extract_shingles
from spark_near_duplicates.similarity import (
    estimate_similarity,
    jaccard_similarity,
)


def test_finds_duplicate_documents_end_to_end():
  documents = {
    'a': 'The room was clean and comfortable!',
    'b': 'the room was clean and comfortable',
    'c': 'Excellent pizza and friendly service',
  }

  matches = find_near_duplicates_locally(
    documents, shingle_size=5, num_hashes=64, num_bands=16,
    similarity_threshold=0.8,
  )

  assert len( matches ) == 1

  left_id, right_id, estimated, exact = matches[ 0 ]

  assert ( left_id, right_id ) == ( 'a', 'b' )
  assert estimated == 1.0
  assert exact == 1.0

def test_does_not_emit_self_or_reversed_pairs():
  documents = {
    'a': 'the room was clean and comfortable',
    'b': 'the room was clean and comfortable',
  }

  matches = find_near_duplicates_locally(
    documents, shingle_size=5, num_hashes=64, num_bands=16,
    similarity_threshold=0.8,
  )

  pairs = { ( left_id, right_id ) for left_id, right_id, _, _ in matches }

  assert pairs == {( 'a', 'b' )}
  assert ( 'a', 'a' ) not in pairs
  assert ( 'b', 'b' ) not in pairs
  assert ( 'b', 'a' ) not in pairs

def find_near_duplicates_locally(
  documents: dict[ str, str ], *, shingle_size: int, num_hashes: int,
  num_bands: int, similarity_threshold: float,
) -> list[ tuple[ str, str, float, float ] ]:
  shingles_by_id = {
    document_id: extract_shingles(
      normalize_text( text ),
      size=shingle_size
    )
    for document_id, text in documents.items()
  }

  signatures_by_id = {
    document_id: minhash_signature(
      shingles,
      num_hashes=num_hashes,
      seed=42
    )
    for document_id, shingles in shingles_by_id.items()
  }

  band_records = [
    record
    for document_id, signature in signatures_by_id.items()
    for record in band_signature(
      document_id,
      signature,
      num_bands=num_bands
    )
  ]

  candidate_pairs = generate_candidate_pairs( band_records )
  matches = []

  for left_id, right_id in sorted(candidate_pairs):
    estimated_similarity = estimate_similarity(
      signatures_by_id[left_id], signatures_by_id[right_id]
    )

    exact_similarity = jaccard_similarity(
      shingles_by_id[left_id], shingles_by_id[right_id]
    )

    if exact_similarity >= similarity_threshold:
      matches.append(( left_id, right_id, estimated_similarity,
                       exact_similarity ))

  return matches
