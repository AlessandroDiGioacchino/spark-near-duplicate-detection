
import re
import string


def normalize_text(
  text: str, *,
  lowercase: bool=True,
  remove_whitespace: bool=True,
  remove_punctuation: bool=True,
  remove_numbers: bool=True ) -> str:

  if not isinstance( text, str ):
    raise TypeError

  if lowercase:
    text = text.lower()

  if remove_punctuation:
    text = text.translate( str.maketrans( '', '', string.punctuation ) )

  if remove_numbers:
    text = re.sub( r'\d+', '', text )

  if remove_whitespace:
    text = ' '.join( text.split() )

  return text
