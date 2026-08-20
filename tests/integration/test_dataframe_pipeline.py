
import pytest

pyspark = pytest.importorskip( 'pyspark' )

from pyspark.sql import DataFrame, SparkSession

from spark_near_duplicates.pipeline import find_near_duplicates


@pytest.fixture( scope='module' )
def spark():
  session = (
    SparkSession.builder
    .master( 'local[2]' )
    .appName( 'near-duplicate-pipeline-tests' )
    .getOrCreate()
  )

  yield session
  session.stop()


def test_pipeline_is_lazy_and_finds_duplicates( spark, monkeypatch ):
  documents = spark.createDataFrame([
    ( 'a', 'The room was clean and comfortable!' ),
    ( 'b', 'the room was clean and comfortable' ),
    ( 'c', 'Excellent pizza and friendly service' ),
  ], [ 'document_id', 'text' ])

  def reject_driver_collection(*args, **kwargs):
    raise AssertionError( 'the pipeline must not call DataFrame.collect()' )

  with monkeypatch.context() as context:
    context.setattr( DataFrame, 'collect', reject_driver_collection )
    matches = find_near_duplicates(
      documents,
      shingle_size=5,
      num_hashes=64,
      num_bands=16,
      similarity_threshold=0.8,
    )

  rows = matches.collect()

  assert len( rows ) == 1
  assert ( rows[ 0 ].left_id, rows[ 0 ].right_id ) == ( 'a', 'b' )
  assert rows[ 0 ].estimated_similarity == 1.0
  assert rows[ 0 ].exact_similarity == 1.0

def test_pipeline_emits_each_pair_once( spark ):
  documents = spark.createDataFrame([
    ( 'a', 'same document text' ),
    ( 'b', 'same document text' ),
    ( 'c', 'same document text' ),
  ], [ 'document_id', 'text' ])

  matches = find_near_duplicates(
    documents,
    shingle_size=3,
    num_hashes=16,
    num_bands=4,
    similarity_threshold=1.0,
  )

  pairs = { ( row.left_id, row.right_id ) for row in matches.collect() }
  assert pairs == {( 'a', 'b' ), ( 'a', 'c' ), ( 'b', 'c' )}

def test_rejects_incompatible_band_configuration( spark ):
  documents = spark.createDataFrame([( 'a', 'text' )],
                                    [ 'document_id', 'text' ])

  with pytest.raises( ValueError ):
    find_near_duplicates( documents, num_hashes=10, num_bands=3 )
