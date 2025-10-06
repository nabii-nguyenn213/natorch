import numpy as np 
import re

def word_tokenize(text : str) -> list[str]:
    text = text.lower()
    text = text.replace("'", "")  
    text = re.sub(r"([.,!?;:()\"“”‘’`])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"\w+|[.,!?;:()\"“”‘’`]", text)
    return tokens
