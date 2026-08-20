
from __future__ import annotations

from typing import TYPE_CHECKING

from spark_near_duplicates.minhash import minhash_signature
from spark_near_duplicates.preprocessing import normalize_text
from spark_near_duplicates.shingles import extract_shingles

if TYPE_CHECKING:
  from pyspark.sql import DataFrame


def find_near_duplicates(
  documents: 'DataFrame', *,
  id_column: str='document_id', text_column: str='text', shingle_size: int=5,
  num_hashes: int=128, num_bands: int=32, similarity_threshold: float=0.8,
  seed: int=42,
) -> 'DataFrame':
  '''Build a lazy, distributed near-duplicate detection pipeline.

  The returned DataFrame has ``left_id``, ``right_id``,
  ``estimated_similarity`` and ``exact_similarity`` columns.  No Spark action
  is called here: candidate generation and scoring remain on the executors.
  Document identifiers are expected to be unique and non-null.
  '''

  from pyspark.sql import functions as F
  from pyspark.sql.types import ArrayType, LongType, StringType

  _validate_arguments(
    documents.columns,
    id_column=id_column,
    text_column=text_column,
    shingle_size=shingle_size,
    num_hashes=num_hashes,
    num_bands=num_bands,
    similarity_threshold=similarity_threshold,
    seed=seed,
  )

  @F.udf(
    returnType=ArrayType( StringType(), containsNull=False ),
    useArrow=True
  )
  def make_shingles( text: str | None ) -> list[ str ]:
    if text is None:
      return []

    return sorted(
      extract_shingles( normalize_text( text ), size=shingle_size )
    )

  @F.udf(
    returnType=ArrayType( LongType(), containsNull=False ),
    useArrow=True
  )
  def make_signature( shingles: list[ str ] ) -> list[ int ]:
    # Spark LongType is signed: mapping unsigned hashes into the signed range
    # preserves equality, which is all MinHash and LSH comparisons require
    return [
      value if value < 2**63 else value - 2**64
      for value in minhash_signature(
        set( shingles ), num_hashes=num_hashes, seed=seed
      )
    ]

  features = (
    documents.select(
      F.col( id_column ).cast( 'string' ).alias('document_id'),
      make_shingles( F.col( text_column ) ).alias( 'shingles' ),
    )
    .where( F.col( 'document_id' ).isNotNull() )
    .where( F.size( 'shingles' ) > 0 )
    .withColumn( 'signature', make_signature( F.col( 'shingles' ) ) )
  )

  rows_per_band = num_hashes // num_bands
  bands = (
    features.select(
      'document_id',
      F.posexplode(
        F.array( *[
          F.slice( 'signature', index * rows_per_band + 1, rows_per_band )
          for index in range( num_bands )
        ] )
      ).alias( 'band_index', 'band' )
    )
    .withColumn(
      'bucket_id',
      F.sha2( F.to_json( F.col( 'band' ) ), 256 )
    )
    .drop( 'band' )
  )

  left_band = bands.alias( 'left_band' )
  right_band = bands.alias( 'right_band' )
  candidate_pairs = (
    left_band.join(
      right_band,
      ( F.col( 'left_band.band_index' ) == F.col( 'right_band.band_index' ) )
      & ( F.col( 'left_band.bucket_id' ) == F.col( 'right_band.bucket_id' ) )
      & ( F.col( 'left_band.document_id' ) < F.col( 'right_band.document_id' ) ),
      'inner'
    )
    .select(
      F.col( 'left_band.document_id' ).alias( 'left_id' ),
      F.col( 'right_band.document_id' ).alias( 'right_id' )
    )
    .distinct()
  )

  left_features = features.select(
    F.col( 'document_id' ).alias( 'left_id' ),
    F.col( 'shingles' ).alias( 'left_shingles' ),
    F.col( 'signature' ).alias( 'left_signature' ),
  )

  right_features = features.select(
    F.col( 'document_id' ).alias( 'right_id' ),
    F.col( 'shingles' ).alias( 'right_shingles' ),
    F.col( 'signature' ).alias( 'right_signature' ),
  )

  scored = (
    candidate_pairs
    .join( left_features, 'left_id' )
    .join( right_features, 'right_id' )
    .withColumn(
      'estimated_similarity',
      F.aggregate(
        F.zip_with(
          'left_signature', 'right_signature',
          lambda left, right: ( left == right ).cast( 'int' )
        ),
        F.lit( 0 ),
        lambda total, matches: total + matches,
      ) / F.size( 'left_signature' ),
    )
    .withColumn(
      'exact_similarity',
      F.size( F.array_intersect( 'left_shingles', 'right_shingles' ) )
      / F.size( F.array_union( 'left_shingles', 'right_shingles' ) ),
    )
  )

  return (
    scored
    .where( F.col( 'exact_similarity' ) >= similarity_threshold )
    .select(
      'left_id', 'right_id', 'estimated_similarity', 'exact_similarity'
    )
  )


def _validate_arguments(
  columns: list[str], *, id_column: str, text_column: str, shingle_size: int,
  num_hashes: int, num_bands: int, similarity_threshold: float, seed: int
) -> None:
  missing = { id_column, text_column }.difference( columns )
  if missing:
    raise ValueError( f'missing required columns: {sorted( missing )}' )
  if not isinstance( shingle_size, int ) or shingle_size <= 0:
    raise ValueError( 'shingle_size must be a positive integer' )
  if not isinstance( num_hashes, int ) or num_hashes <= 0:
    raise ValueError( 'num_hashes must be a positive integer' )
  if not isinstance( num_bands, int ) or num_bands <= 0:
    raise ValueError( 'num_bands must be a positive integer' )
  if num_hashes % num_bands != 0:
    raise ValueError( 'num_hashes must be divisible by num_bands' )
  if not isinstance( similarity_threshold, ( int, float ) ):
    raise TypeError( 'similarity_threshold must be numeric' )
  if not 0 <= similarity_threshold <= 1:
    raise ValueError( 'similarity_threshold must be between 0 and 1' )
  if not isinstance( seed, int ) or not 0 <= seed < 2**64:
    raise ValueError( 'seed must be an integer between 0 and 2**64 - 1' )
