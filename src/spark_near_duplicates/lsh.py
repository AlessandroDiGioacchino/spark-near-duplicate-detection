
import hashlib

from collections import defaultdict
from itertools import combinations
from collections.abc import Iterable


BandKey = tuple[ int, str ]
BandRecord = tuple[ BandKey, str ]
DocumentPair = tuple[ str, str ]


def _hash_band( values: tuple[ int, ... ] ) -> str:
  hasher = hashlib.sha256()

  for value in values:
    if not isinstance( value, int ):
      raise TypeError( 'signature value must be integers' )

    if not 0 <= value < 2**64:
      raise ValueError( 'signature value must be between 0 and 2**64 - 1' )

    hasher.update( value.to_bytes( 8, byteorder='big', signed=False ) )

  return hasher.hexdigest()


def band_signature(
    document_id: str,
    signature: tuple[ int, ... ],
    *,
    num_bands: int,
) -> list[ tuple[ tuple[ int, int ], str ] ]:
  if not isinstance( document_id, str ):
    raise TypeError( 'document_id must be a string' )

  if not isinstance( num_bands, int ):
    raise TypeError( 'num_bands must be an integer' )

  if num_bands <= 0:
    raise ValueError( 'num_bands must be positive' )

  if not signature:
    raise ValueError( 'signature must not be empty' )

  if len( signature ) % num_bands != 0:
    raise ValueError( 'signature length must be divisible by num_bands' )

  rows_per_band = len( signature ) // num_bands
  records = []

  for band_index in range( num_bands ):
    start = band_index * rows_per_band
    end = start + rows_per_band
    band = signature[ start : end ]

    bucket_id = _hash_band( band )
    key = ( band_index, bucket_id )

    records.append(( key, document_id ))

  return records


def generate_candidate_pairs(
  records: Iterable[ BandRecord ]
) -> set[ DocumentPair ]:

  buckets: dict[ BandKey, set[ str ] ] = defaultdict( set )
  for key, document_id in records:
    buckets[ key ].add( document_id )

  candidate_pairs: set[ DocumentPair ] = set()
  for document_ids in buckets.values():
    if len( document_ids ) < 2:
      continue

    ordered_ids = sorted( document_ids )
    candidate_pairs.update( combinations( ordered_ids, 2 ) )

  return candidate_pairs
