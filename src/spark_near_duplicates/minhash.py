
import hashlib


def _hash_shingle( shingle: str, *, seed: int, hash_index: int ) -> int:
  '''Hash one shingle for one member of the MinHash family.'''
  hasher = hashlib.sha256()

  # Fixed-width integers avoid ambiguous inputs such as (1, 23) and (12, 3).
  hasher.update( seed.to_bytes( 8, byteorder='big', signed=False ) )
  hasher.update( hash_index.to_bytes( 8, byteorder='big', signed=False ))
  hasher.update( shingle.encode( 'utf-8' ))

  # Eight bytes give us a stable unsigned 64-bit value.
  return int.from_bytes( hasher.digest()[ :8 ], byteorder='big', signed=False )

def minhash_signature(
  shingles: set[ str ], *,
  num_hashes: int, seed: int
) -> tuple[ int, ... ]:

  if not isinstance( num_hashes, int ):
    raise TypeError( 'num_hashes must be an integer' )

  if num_hashes <= 0:
    raise ValueError( 'num_hashes must be positive' )

  if not isinstance( seed, int ):
    raise TypeError( 'seed must be an integer' )

  if not 0 <= seed < 2**64:
    raise ValueError( 'seed must be between 0 and 2**64 - 1' )

  if not shingles:
    raise ValueError( 'cannot create a MinHash signature from no shingles' )

  if any( not isinstance( shingle, str ) for shingle in shingles ):
    raise TypeError( 'all shingles must be strings' )

  return tuple(
    min(
      _hash_shingle(
        shingle,
        seed=seed,
        hash_index=hash_index,
      )
      for shingle in shingles
    )
    for hash_index in range( num_hashes )
  )
