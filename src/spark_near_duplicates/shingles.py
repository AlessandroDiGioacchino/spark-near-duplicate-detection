
def extract_shingles( text: str, size: int ) -> set[ str ]:
  if size <= 0:
    raise ValueError

  shingles = set()

  for i in range( len( text ) - size + 1 ):
    shingle = text[ i : i + size ]
    shingles.add( shingle )

  return shingles
