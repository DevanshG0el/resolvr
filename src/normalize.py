"""Record normalisation.

Every record gets split into comparable fields instead of being "cleaned away":
  name_full / name_core (legal forms removed) / name_sorted / name_nospace / name_phon /
  legal (canonical legal-form set) / web_stem / name_nums / native flag,
  addr_clean (components kept, abbreviations expanded) / addr_str / addr_tok / addr_nums / state,
  emb_text (lightly cleaned text for the neural retriever).

Indic-script text is transliterated with a token dictionary mined from training matches
(native token -> Latin token that co-occurs with it in the matched Source 1 record),
falling back to anyascii.
"""
import re
import unicodedata

import polars as pl

INDIC_PL = r"[\x{0900}-\x{0DFF}]"
INDIC_RE = re.compile(r"[ऀ-෿]")
TOKEN_SPLIT_RE = re.compile(r"(\s+|,)")

# ----------------------------------------------------------------------------- dictionaries
LEGAL_MAP = {
    "private": "private", "pvt": "private", "prv": "private", "pte": "private", "pvtltd": "private",
    "limited": "limited", "ltd": "limited", "ltda": "limited",
    "llc": "llc", "llp": "llp", "lllp": "llp", "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cos": "co",
    "plc": "plc", "pllc": "pllc", "pc": "pc", "lp": "lp", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "scm": "scm", "scp": "scp", "gie": "gie",
    "ei": "ei", "eirl": "eirl",  # French sole-trader forms (Entreprise Individuelle)
}
FR_NAME_ABBR = {"frs": "freres"}  # French name abbreviations that occur as noise in S2/S3 (France only)
LEGAL_CANON = sorted(set(LEGAL_MAP.values()))
HONORIFICS = ["mr", "mrs", "ms", "messrs", "the"]

GEN_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive",
    "blvd": "boulevard", "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ter": "terrace", "trl": "trail", "sq": "square",
    "pt": "point", "mt": "mount", "ft": "fort", "n": "north", "s": "south", "e": "east",
    "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor", "bldg": "building",
    "rm": "room", "nr": "near", "opp": "opposite", "sec": "sector", "ngr": "nagar",
    "dist": "district", "tq": "taluka", "tal": "taluka", "hno": "house", "expy": "expressway",
    "fwy": "freeway", "crk": "creek", "cv": "cove", "xing": "crossing", "hts": "heights",
    "jn": "junction", "jct": "junction", "extn": "extension", "ext": "extension", "colny": "colony",
}
FR_ABBR = {
    "r": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard", "av": "avenue",
    "ave": "avenue", "pl": "place", "imp": "impasse", "ch": "chemin", "chem": "chemin",
    "rte": "route", "all": "allee", "sq": "square", "st": "saint", "ste": "sainte",
    "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "crs": "cours",
    "res": "residence", "lot": "lotissement", "cte": "cote", "mte": "montee",
    "prom": "promenade", "sent": "sentier", "hab": "habitation", "bat": "batiment",
}
STREET_WORDS = {
    "street", "road", "avenue", "drive", "boulevard", "lane", "court", "place", "highway",
    "parkway", "circle", "terrace", "trail", "square", "way", "rue", "chemin", "impasse",
    "allee", "route", "marg", "path", "floor", "apartment", "suite", "unit", "building",
    "room", "near", "opposite", "no", "number", "plot", "house", "flat", "box", "po",
}

STOPWORDS = {"of", "the", "and", "la", "le", "les", "de", "du", "des", "d", "l", "et", "en", "sur", "city", "region"}

US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new hampshire",
    "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington",
    "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
    "PR": "puerto rico", "GU": "guam", "VI": "virgin islands",
}
IN_STATES = {
    "AP": ["andhra pradesh"], "AR": ["arunachal pradesh"], "AS": ["assam"], "BR": ["bihar"],
    "CG": ["chhattisgarh", "chattisgarh", "ct"], "GA": ["goa"], "GJ": ["gujarat"],
    "HR": ["haryana"], "HP": ["himachal pradesh"], "JH": ["jharkhand"], "KA": ["karnataka"],
    "KL": ["kerala"], "MP": ["madhya pradesh"], "MH": ["maharashtra"], "MN": ["manipur"],
    "ML": ["meghalaya"], "MZ": ["mizoram"], "NL": ["nagaland"], "OD": ["odisha", "orissa", "or"],
    "PB": ["punjab"], "RJ": ["rajasthan"], "SK": ["sikkim"], "TN": ["tamil nadu", "tamilnadu"],
    "TG": ["telangana", "ts"], "TR": ["tripura"], "UP": ["uttar pradesh"],
    "UK": ["uttarakhand", "uttaranchal", "ut"], "WB": ["west bengal"],
    "DL": ["delhi", "nct of delhi", "national capital territory of delhi"],
    "JK": ["jammu and kashmir", "jammu kashmir"], "LA": ["ladakh"], "CH": ["chandigarh"],
    "PY": ["puducherry", "pondicherry"], "AN": ["andaman and nicobar islands"],
    "DN": ["dadra and nagar haveli and daman and diu", "dadra and nagar haveli", "daman and diu"],
    "LD": ["lakshadweep"],
}
FR_REGIONS = {
    "ARA": ["auvergne rhone alpes"], "BFC": ["bourgogne franche comte"], "BRE": ["bretagne"],
    "CVL": ["centre val de loire", "centre"], "COR": ["corse"], "GES": ["grand est"],
    "HDF": ["hauts de france"], "IDF": ["ile de france"], "NOR": ["normandie"],
    "NAQ": ["nouvelle aquitaine"], "OCC": ["occitanie"], "PDL": ["pays de la loire"],
    "PAC": ["provence alpes cote dazur", "provence alpes cote d azur", "paca"],
    "GUA": ["guadeloupe"], "MTQ": ["martinique"], "GUF": ["guyane"], "LRE": ["la reunion", "reunion"],
    "MAY": ["mayotte"],
}


def simple_norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def _state_alias_map():
    """country -> {normalised alias: canonical code}."""
    us = {}
    for code, name in US_STATES.items():
        us[code.lower()] = code
        us[simple_norm(name)] = code
    ind = {}
    for code, names in IN_STATES.items():
        ind[code.lower()] = code
        for n in names:
            ind[simple_norm(n)] = code
    fr = {}
    for code, names in FR_REGIONS.items():
        for n in names:
            fr[simple_norm(n)] = code
    return {"US": us, "India": ind, "France": fr}


STATE_ALIASES = _state_alias_map()
STATE_NAME = {("US", c): n for c, n in US_STATES.items()}
STATE_NAME.update({("India", c): n[0] for c, n in IN_STATES.items()})
STATE_NAME.update({("France", c): n[0] for c, n in FR_REGIONS.items()})
ALL_STATE_KEYS = sorted({k for m in STATE_ALIASES.values() for k in m})


# ----------------------------------------------------------------------------- transliteration
def _anyascii_token(tok: str) -> str:
    from anyascii import anyascii

    out = anyascii(tok).lower()
    return re.sub(r"m(?=[kgcjtd])", "n", out)  # anusvara before a stop reads as "n"


def _translit_tokens(s: str, tokdict: dict) -> str:
    parts = []
    for tok in TOKEN_SPLIT_RE.split(s):
        if tok and INDIC_RE.search(tok):
            key = tok.strip()
            parts.append(tokdict.get(key) or _anyascii_token(key))
        else:
            parts.append(tok)
    return "".join(parts)


def translit_text(s, tokdict, compdict=None):
    """Replace Indic-script text: whole comma-separated components via compdict (e.g. a native
    state name -> "west bengal"), then single tokens via tokdict, else anyascii."""
    if s is None or not INDIC_RE.search(s):
        return s
    if compdict:
        comps = s.split(",")
        s = ",".join(
            (" " + compdict[c.strip()]) if c.strip() in compdict else c for c in comps
        )
    return _translit_tokens(s, tokdict)


def translit_column(df: pl.DataFrame, col: str, tokdict: dict, compdict: dict | None = None) -> pl.DataFrame:
    """Transliterate only the distinct Indic-containing values of a column, then join back."""
    vals = df.filter(pl.col(col).str.contains(INDIC_PL)).select(pl.col(col).unique())
    if vals.height == 0:
        return df
    mapped = vals.with_columns(
        pl.col(col).map_elements(lambda s: translit_text(s, tokdict, compdict), return_dtype=pl.Utf8).alias("_tl")
    )
    return (
        df.join(mapped, on=col, how="left")
        .with_columns(pl.coalesce("_tl", col).alias(col))
        .drop("_tl")
    )


def _best_by_lift(tu: pl.DataFrame, min_count: int, min_prob: float) -> dict:
    """tu: one row per (t, u) co-occurrence. Keep, per native key t, the Latin value u with the
    highest lift P(u|t)/P(u) among those with P(u|t) >= min_prob (frequent words such as
    "private" co-occur with everything, so raw P(u|t) alone picks them)."""
    n_rows = tu.select(pl.col("row").n_unique()).item()
    t_cnt = tu.group_by("t").agg(pl.col("row").n_unique().alias("nt"))
    u_cnt = tu.group_by("u").agg((pl.col("row").n_unique() / n_rows).alias("pu"))
    best = (
        tu.unique(["row", "t", "u"]).group_by(["t", "u"]).len(name="ntu")
        .join(t_cnt, on="t").join(u_cnt, on="u")
        .with_columns((pl.col("ntu") / pl.col("nt")).alias("prob"))
        .with_columns((pl.col("prob") / pl.col("pu")).alias("lift"))
        .filter((pl.col("nt") >= min_count) & (pl.col("prob") >= min_prob))
        .sort(["t", "lift", "ntu"], descending=[False, True, True])
        .group_by("t", maintain_order=True).first()
    )
    return dict(zip(best["t"].to_list(), best["u"].to_list()))


def _clean_tokens(c: str) -> pl.Expr:
    return (
        pl.col(c).fill_null("").str.to_lowercase().str.split(" ")
        .list.eval(pl.element().str.replace_all(r"[,.()\[\]|]", "").filter(pl.element() != "")).list.unique()
    )


def mine_token_dict(pairs: pl.DataFrame, min_count: int = 2, min_prob: float = 0.5) -> dict:
    """pairs: native_text (S2/S3 side), latin_text (matched S1 side). Indic token -> Latin token."""
    p = pairs.select(
        _clean_tokens("native_text").list.eval(pl.element().filter(pl.element().str.contains(INDIC_PL))).alias("t"),
        _clean_tokens("latin_text").list.eval(pl.element().filter(~pl.element().str.contains(INDIC_PL))).alias("u"),
    ).filter(pl.col("t").list.len() > 0).with_row_index("row")
    tu = p.explode("t").explode("u").drop_nulls()
    return _best_by_lift(tu, min_count, min_prob)


def mine_component_dict(pairs: pl.DataFrame, min_count: int = 3, min_prob: float = 0.6) -> dict:
    """pairs: native_text (S2/S3 address), latin_value (matched S1's canonical state name).
    Whole Indic address component -> state name (handles multi-word states)."""
    p = pairs.select(
        pl.col("native_text").fill_null("").str.split(",")
        .list.eval(pl.element().str.strip_chars().filter(pl.element().str.contains(INDIC_PL))).alias("t"),
        pl.col("latin_value").alias("u"),
    ).filter((pl.col("t").list.len() > 0) & pl.col("u").is_not_null()).with_row_index("row")
    tu = p.explode("t").drop_nulls()
    return _best_by_lift(tu, min_count, min_prob)


# ----------------------------------------------------------------------------- expressions
def _ascii_fold(e: pl.Expr) -> pl.Expr:
    return e.str.normalize("NFKD").str.replace_all(r"\p{Mn}", "").str.to_lowercase()


WEB_TLD = r"(?:com|in|org|net|co|fr|biz|info|io|us)"
WEB_STEM_RE = rf"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)(?:\.[a-z0-9-]+)*\.{WEB_TLD}\b"
WEB_FULL_RE = rf"(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*\.{WEB_TLD}\b\S*"


def _phonetic(e: pl.Expr) -> pl.Expr:
    """Crude consonant skeleton per token: merges vowel noise and common transliteration variants."""
    e = (
        e.str.replace_all("ph", "f").str.replace_all("sh", "s").str.replace_all("ck", "k")
        .str.replace_all("q", "k").str.replace_all("z", "s").str.replace_all("v", "w")
        .str.replace_all(r"\B[aeiouyh]+", "")
    )
    for c in "bcdfgjklmnprstwx":  # the regex engine has no backreferences
        e = e.str.replace_all(f"{c}{c}+", c)
    return e


def normalize_names(df: pl.DataFrame, tokdict: dict) -> pl.DataFrame:
    df = df.with_columns(pl.col("name").fill_null("").alias("_n"))
    df = df.with_columns(pl.col("_n").str.contains(INDIC_PL).alias("name_native"))
    df = translit_column(df, "_n", tokdict)
    n = _ascii_fold(pl.col("_n"))
    df = df.with_columns(n.alias("_n"))
    df = df.with_columns(
        pl.col("_n").str.extract(WEB_STEM_RE, 1).str.replace_all("-", "").alias("web_stem"),
        pl.col("_n").str.replace_all(WEB_FULL_RE, " ").alias("_n"),
    )
    cleaned = (
        pl.col("_n").str.replace_all("&", " and ").str.replace_all(r"[.'`’]", "")
        .str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
    )
    df = df.with_columns(cleaned.alias("_n"))
    toks = (
        pl.col("_n").str.replace(r"^m s ", "").str.split(" ")
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.eval(pl.element().replace(LEGAL_MAP))
        .list.eval(pl.element().filter((pl.element() != pl.element().shift(1)).fill_null(True)))
    )
    df = df.with_columns(toks.alias("_t"))
    if "country" in df.columns:
        df = df.with_columns(pl.when(pl.col("country") == "France")
                             .then(pl.col("_t").list.eval(pl.element().replace(FR_NAME_ABBR)))
                             .otherwise(pl.col("_t")).alias("_t"))
    # drop a leading honorific
    df = df.with_columns(
        pl.when(pl.col("_t").list.first().is_in(HONORIFICS) & (pl.col("_t").list.len() > 1))
        .then(pl.col("_t").list.slice(1)).otherwise(pl.col("_t")).alias("_t")
    )
    core = pl.col("_t").list.eval(pl.element().filter(~pl.element().is_in(LEGAL_CANON)))
    df = df.with_columns(
        pl.col("_t").list.eval(pl.element().filter(pl.element().is_in(LEGAL_CANON))).list.unique().list.sort()
        .list.join(" ").alias("legal"),
        core.alias("_core"),
    )
    df = df.with_columns(
        pl.when(pl.col("_core").list.len() == 0)
        .then(pl.when(pl.col("web_stem").is_not_null()).then(pl.concat_list(pl.col("web_stem"))).otherwise(pl.col("_t")))
        .otherwise(pl.col("_core")).alias("_core")
    )
    df = df.with_columns(
        pl.when(pl.col("_t").list.len() > 0).then(pl.col("_t")).otherwise(pl.col("_core"))
        .list.join(" ").alias("name_full"),
        pl.col("_core").list.join(" ").alias("name_core"),
        pl.col("_core").list.sort().list.join(" ").alias("name_sorted"),
        pl.col("_core").list.join("").alias("name_nospace"),
        pl.col("_core").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).alias("name_tok"),
        pl.col("_n").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("name_nums"),
    )
    df = df.with_columns(
        _phonetic(pl.col("name_core")).alias("name_phon"),
        pl.col("name_tok").list.unique().alias("name_tok"),
    )
    return df.drop(["_n", "_t", "_core"])


def _abbr_expand(e: pl.Expr, mapping: dict) -> pl.Expr:
    pats = [f" {k} " for k in mapping]
    reps = [f" {v} " for v in mapping.values()]
    return (pl.lit(" ") + e + pl.lit(" ")).str.replace_many(pats, reps).str.replace_many(pats, reps).str.strip_chars()


def normalize_addresses(df: pl.DataFrame, tokdict: dict, compdict: dict | None = None) -> pl.DataFrame:
    df = df.with_columns(pl.col("address").fill_null("").alias("_a"))
    df = translit_column(df, "_a", tokdict, compdict)
    a = (
        _ascii_fold(pl.col("_a"))
        .str.replace_all(r"<\s*null\s*>|\bnull\b|\bnone\b|\bn/a\b|\bnan\b", " ")
        .str.replace_all(r"[.'`’#]", "")
        .str.replace_all(r"[^\p{L}\p{N},]+", " ")
        .str.replace_all(r"\s*,\s*", " , ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars(" ,")
    )
    df = df.with_columns(a.alias("_a"))
    df = df.with_columns(
        pl.when(pl.col("country") == "France")
        .then(_abbr_expand(pl.col("_a"), FR_ABBR))
        .otherwise(_abbr_expand(pl.col("_a"), GEN_ABBR)).alias("_a")
    )
    comps = pl.col("_a").str.split(" , ").list.eval(
        pl.element().str.strip_chars(" ,").filter(pl.element() != "")
    )
    df = df.with_columns(comps.alias("_c"))
    state = pl.lit(None, dtype=pl.Utf8)
    for country, amap in STATE_ALIASES.items():
        st = (
            pl.col("_c").list.eval(
                pl.element().replace_strict(list(amap), list(amap.values()), default=None, return_dtype=pl.Utf8)
            ).list.drop_nulls().list.last()
        )
        state = pl.when(pl.col("country") == country).then(st).otherwise(state)
    df = df.with_columns(state.alias("state"))
    nostate = pl.col("_c").list.eval(pl.element().filter(~pl.element().is_in(ALL_STATE_KEYS)))
    df = df.with_columns(nostate.alias("_c2"))
    df = df.with_columns(
        pl.col("_c").list.join(", ").alias("addr_clean"),
        pl.col("_c2").list.join(" ").alias("addr_str"),
        pl.col("_c2").list.join(" ").str.split(" ")
        .list.eval(pl.element().filter((pl.element().str.len_chars() > 1) & ~pl.element().str.contains(r"^\d+$")))
        .list.unique().alias("addr_tok"),
        pl.col("_a").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("addr_nums"),
        pl.col("_a").str.extract(r"(\d+)", 1).str.strip_chars_start("0").alias("addr_num1"),
    )
    df = df.with_columns(
        pl.col("addr_tok").list.eval(pl.element().filter(~pl.element().is_in(list(STREET_WORDS | STOPWORDS)))).alias("addr_loc_tok"),
        (pl.col("addr_str").str.len_chars() == 0).alias("addr_empty"),
    )
    return df.drop(["_a", "_c", "_c2"])


def normalize(df: pl.DataFrame, dicts: dict) -> pl.DataFrame:
    """df columns: id, src, name, address, country. dicts: name / addr / addr_comp mined mappings."""
    df = normalize_names(df, dicts.get("name", {}))
    df = normalize_addresses(df, dicts.get("addr", {}), dicts.get("addr_comp", {}))
    return df.with_columns(
        (pl.lit("query: ") + pl.col("name_full") + pl.lit(" , ") + pl.col("addr_clean")).alias("emb_text")
    )
