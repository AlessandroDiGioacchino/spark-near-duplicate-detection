
def jaccard_similarity( left: set[ str ], right: set[ str ] ) -> float:
  '''Compute Jaccard similarity of two sets'''
  intersection_length = len( left.intersection( right ) )
  union_length = len( left.union( right ) )

  if union_length == 0:
    return 1

  return float( intersection_length ) / float( union_length )

def estimate_similarity( left: tuple[ int, ... ],
                         right: tuple[ int, ... ] ) -> float:
  '''Estimate Jaccard similarity from two MinHash signatures'''

  if len( left ) != len( right ):
    raise ValueError( 'signatures must have equal length' )

  if not left or not right:
    raise ValueError( 'signatures must not be empty' )

  matching_positions = sum(
    left_value == right_value
    for left_value, right_value in zip( left, right )
  )

  return matching_positions / len( left )
