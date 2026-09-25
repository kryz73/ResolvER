"""
ML Challenge 2026 — Business Entity Resolution
Data Preprocessing, Transliteration & Normalization
"""

import re
import unicodedata
import pandas as pd
from typing import Tuple, Dict, Set, Optional

# Indic Unicode script ranges
INDIC_REGEX = re.compile(r'[\u0900-\u0D7F]')

# Common Legal Suffixes Mappings
US_LEGAL_MAP = {
    r'\bincorporated\b': 'inc',
    r'\bcorporation\b': 'corp',
    r'\bcompany\b': 'co',
    r'\blimited liability company\b': 'llc',
    r'\blimited liability partnership\b': 'llp',
    r'\blimited\b': 'ltd',
    r'\benterprises\b': 'ent',
    r'\bassociates\b': 'assoc',
}

INDIA_LEGAL_MAP = {
    r'\bprivate limited\b': 'pvt ltd',
    r'\bpvt\.?\s*ltd\.?\b': 'pvt ltd',
    r'\bp\.?\s*ltd\.?\b': 'pvt ltd',
    r'\bprivate\b': 'pvt',
    r'\blimited\b': 'ltd',
    r'\bllp\b': 'llp',
    r'\benterprises\b': 'ent',
    r'\btrading\b': 'trd',
    r'\bcompany\b': 'co',
}

FRANCE_LEGAL_MAP = {
    r'\bsociete a responsabilite limitee\b': 'sarl',
    r'\bsociete par actions simplifiee unipersonnelle\b': 'sasu',
    r'\bsociete par actions simplifiee\b': 'sas',
    r'\bentreprise unipersonnelle a responsabilite limitee\b': 'eurl',
    r'\bsociete civile immobiliere\b': 'sci',
    r'\bsociete anonyme\b': 'sa',
    r'\bassociation\b': 'assoc',
}

# French Street Normalizations
FRANCE_STREET_MAP = {
    r'\bboulevard\b': 'bd',
    r'\bbvd\b': 'bd',
    r'\bavenue\b': 'av',
    r'\bimpasse\b': 'imp',
    r'\bchemin\b': 'che',
    r'\brue\b': 'rue',
    r'\ballee\b': 'all',
}

# US Street Normalizations
US_STREET_MAP = {
    r'\bstreet\b': 'st',
    r'\broad\b': 'rd',
    r'\bavenue\b': 'ave',
    r'\bboulevard\b': 'blvd',
    r'\bdrive\b': 'dr',
    r'\blane\b': 'ln',
    r'\bcourt\b': 'ct',
    r'\bhighway\b': 'hwy',
    r'\bsuite\b': 'ste',
    r'\bapartment\b': 'apt',
}

# Fast Indic transliteration fallback map (Devanagari to Latin approximation)
DEVANAGARI_MAP = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'क': 'k', 'ख': 'kh', 'ग': 'g',
    'घ': 'gh', 'ङ': 'ng', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n', 'त': 't', 'थ': 'th',
    'द': 'd', 'ध': 'dh', 'न': 'n', 'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh',
    'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh',
    'स': 's', 'ह': 'h', 'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n', '्': '',
    'ः': 'h', '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5',
    '६': '6', '७': '7', '८': '8', '९': '9'
}


def basic_clean_text(text: Optional[str]) -> str:
    """Standardizes casing, strips noise symbols, and collapses whitespace."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    # Strip URL schemes
    text = re.sub(r'https?://(?:www\.)?', '', text, flags=re.IGNORECASE)
    # Strip leading noise symbols (-- , ##, <<)
    text = re.sub(r'^[\-\#\<\>\*\@\.\s]+', '', text)
    # Replace ampersands with 'and'
    text = re.sub(r'\s*&\s*', ' and ', text)
    # Normalize unicode to NFKC
    text = unicodedata.normalize('NFKC', text)
    # Lowercase and normalize whitespace
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    return text


def transliterate_indic(text: str) -> str:
    """Converts Indic scripts into Romanized Latin text using transliteration."""
    if not text or not INDIC_REGEX.search(text):
        return text

    # Attempt to use indic_transliteration if available
    try:
        from indic_transliteration import sanscript
        # Detect Devanagari
        if re.search(r'[\u0900-\u097F]', text):
            roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
            return roman.lower()
        elif re.search(r'[\u0D00-\u0D7F]', text):
            roman = sanscript.transliterate(text, sanscript.MALAYALAM, sanscript.ITRANS)
            return roman.lower()
        elif re.search(r'[\u0B80-\u0BFF]', text):
            roman = sanscript.transliterate(text, sanscript.TAMIL, sanscript.ITRANS)
            return roman.lower()
        elif re.search(r'[\u0C00-\u0C7F]', text):
            roman = sanscript.transliterate(text, sanscript.TELUGU, sanscript.ITRANS)
            return roman.lower()
        elif re.search(r'[\u0C80-\u0CFF]', text):
            roman = sanscript.transliterate(text, sanscript.KANNADA, sanscript.ITRANS)
            return roman.lower()
        elif re.search(r'[\u0980-\u09FF]', text):
            roman = sanscript.transliterate(text, sanscript.BENGALI, sanscript.ITRANS)
            return roman.lower()
        elif re.search(r'[\u0A80-\u0AFF]', text):
            roman = sanscript.transliterate(text, sanscript.GUJARATI, sanscript.ITRANS)
            return roman.lower()
    except (ImportError, Exception):
        pass

    # Fast character-level fallback mapping for Devanagari
    chars = []
    for ch in text:
        chars.append(DEVANAGARI_MAP.get(ch, ch))
    return "".join(chars)


def standardize_name(name: str, country: str) -> Tuple[str, str]:
    """
    Standardizes a business name:
    Returns (normalized_name, transliterated_name)
    """
    cleaned = basic_clean_text(name)
    trans = transliterate_indic(cleaned) if country == "India" else cleaned

    # Apply country legal mappings to transliterated text
    if country == "US":
        for pat, repl in US_LEGAL_MAP.items():
            trans = re.sub(pat, repl, trans)
    elif country == "India":
        for pat, repl in INDIA_LEGAL_MAP.items():
            trans = re.sub(pat, repl, trans)
    elif country == "France":
        for pat, repl in FRANCE_LEGAL_MAP.items():
            trans = re.sub(pat, repl, trans)

    # Clean domain extensions (.com, .org, .in, .fr, .net)
    trans = re.sub(r'\.(?:com|org|net|in|fr|co|gov|edu)\b', '', trans)
    trans = re.sub(r'[^a-z0-9\s]', ' ', trans)
    trans = re.sub(r'\s+', ' ', trans).strip()

    return cleaned, trans


def extract_numbers_and_postal(address: str, country: str) -> Dict:
    """Extracts numeric building tokens and country-specific postal codes."""
    if not address or pd.isna(address):
        return {'numbers': set(), 'postal': ''}

    s_addr = str(address)
    # Extract all digit clusters
    numbers = set(re.findall(r'\b\d+\b', s_addr))

    postal = ''
    if country == "US":
        m = re.search(r'\b\d{5}(?:-\d{4})?\b', s_addr)
        if m: postal = m.group(0)[:5]
    elif country == "India":
        m = re.search(r'\b[1-9]\d{5}\b', s_addr)
        if m: postal = m.group(0)
    elif country == "France":
        m = re.search(r'\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b', s_addr)
        if m: postal = m.group(0)

    return {'numbers': numbers, 'postal': postal}


def standardize_address(address: str, country: str) -> str:
    """Standardizes address text, applying abbreviation expansions and cleaning."""
    if not address or pd.isna(address) or str(address).strip().lower() == "nan":
        return ""

    cleaned = basic_clean_text(address)
    if country == "India":
        cleaned = transliterate_indic(cleaned)
        # Clean Indian landmark prefixes
        cleaned = re.sub(r'\b(?:near|opp|opposite|behind|beside|above|below)\b', '', cleaned)

    if country == "US":
        for pat, repl in US_STREET_MAP.items():
            cleaned = re.sub(pat, repl, cleaned)
    elif country == "France":
        for pat, repl in FRANCE_STREET_MAP.items():
            cleaned = re.sub(pat, repl, cleaned)

    cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def preprocess_dataframe(df: pd.DataFrame, is_reference: bool = False) -> pd.DataFrame:
    """
    Applies high-speed preprocessing pipeline to a source dataframe using fast zip iteration:
    Adds:
      - 'clean_name': cleaned lowercase string
      - 'norm_name': transliterated, legal-standardized Latin string
      - 'norm_address': standardized address string
      - 'postal_code': extracted postal code (if found)
      - 'address_missing': boolean flag for missing address
    """
    df = df.copy()
    names = df['business_name'].fillna('').astype(str).tolist()
    addrs = df['business_address'].fillna('').astype(str).tolist()
    countries = df['country'].fillna('US').astype(str).tolist()

    clean_names = []
    norm_names = []
    norm_addrs = []
    postals = []
    missing_flags = []

    for raw_name, raw_addr, c in zip(names, addrs, countries):
        is_missing = (not raw_addr) or (raw_addr.strip() == '') or (raw_addr.strip().lower() == 'nan')
        missing_flags.append(is_missing)

        c_name, n_name = standardize_name(raw_name, c)
        n_addr = standardize_address(raw_addr, c)
        p_info = extract_numbers_and_postal(raw_addr, c)

        clean_names.append(c_name)
        norm_names.append(n_name)
        norm_addrs.append(n_addr)
        postals.append(p_info['postal'])

    df['address_missing'] = missing_flags
    df['clean_name'] = clean_names
    df['norm_name'] = norm_names
    df['norm_address'] = norm_addrs
    df['postal_code'] = postals

    return df

