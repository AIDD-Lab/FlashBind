import time, json
import re
import xxhash
from itertools import islice
from Bio import Entrez, SeqIO
import pandas as pd
import requests
import logging
import os
Entrez.email = os.environ["ENTREZ_EMAIL"]
Entrez.api_key = os.environ["ENTREZ_API_KEY"]

FIELD_SPLITTER = re.compile(r"\s*,\s*")

def chunks(iterable, n):
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch

def split_field(s: str) -> list[str]:
    """Split comma-separated field string"""
    if not s or pd.isna(s):
        return []
    return [x for x in FIELD_SPLITTER.split(str(s).strip()) if x]

def hash_sequence(sequence: str) -> str:
    """Use xxhash to hash the sequence"""
    return xxhash.xxh64(sequence.encode('utf-8')).hexdigest()

def safe_entrez_read(func, max_retries=3, sleep_base=1.0, **kwargs):
    for attempt in range(max_retries):
        try:
            handle = func(**kwargs)
            
            # Determine how to handle the result based on rettype and retmode
            rettype = kwargs.get("rettype", "")
            retmode = kwargs.get("retmode", "")
            if retmode == "json":
                # Read raw content and handle bytes/encoding properly
                raw_content = handle.read()
                if isinstance(raw_content, bytes):
                    content = raw_content.decode('utf-8')
                else:
                    content = raw_content
                # Clean invalid control characters before parsing JSON
                import re
                # Replace problematic characters in JSON strings
                cleaned_content = content.strip()
                # Replace unescaped tabs, newlines, and other control chars with escaped versions
                cleaned_content = cleaned_content.replace('\t', '\\t')
                cleaned_content = cleaned_content.replace('\n', '\\n')
                cleaned_content = cleaned_content.replace('\r', '\\r')
                # Remove other invalid control characters
                cleaned_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned_content)
                # Try to parse JSON
                try:
                    result = json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at position {e.pos}")
                    print(f"Content around error (±50 chars): '{cleaned_content[max(0, e.pos-50):e.pos+50]}'")
                    print(f"Character at error position: {repr(cleaned_content[e.pos] if e.pos < len(cleaned_content) else 'EOF')}")
                    raise
            elif rettype == "fasta":
                # For fasta format, return SeqIO parser result list
                result = list(SeqIO.parse(handle, "fasta"))
            else:
                # Default use Entrez.read
                result = Entrez.read(handle)
            handle.close()
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            exit()
            print(f"[WARN] Entrez call failed on {e}")
            if attempt < max_retries - 1:
                wait = sleep_base * (2 ** attempt)  
                print(f"[INFO] Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] Permanent failure on {e}")
                raise

def retry_request(func, max_retries=3, delay=1.0, backoff=2.0, **kwargs):
    """
    Retry a requests function with exponential backoff.
    """
    for attempt in range(max_retries + 1):
        try:
            response = func(**kwargs)
            return response
        except requests.RequestException as e:
            if attempt == max_retries:
                logging.warning(f"Request failed after {max_retries} retries: {e}")
                return None
            else:
                wait_time = delay * (backoff ** attempt)
                logging.debug(f"Request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
    
    return None