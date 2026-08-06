"""
Utilities for the biological validation section of the STRING PPI analysis.

Requires: pip install goatools requests pandas

Downloads (all cached after first run):
  - STRING protein.info (for STRING ID -> gene symbol mapping)
  - NCBI gene2go + gene_info (for gene symbol -> GO term associations)
  - go-basic.obo (reused from go_dag/data/ if present, else downloaded fresh)
"""
import os
import gzip
import time
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROTEIN_INFO_URL = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
GENE2GO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
GO_OBO_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"

HUMAN_TAXID = "9606"


def _make_session(max_retries=5):
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _download(url, local_path, max_retries=5, chunk_size=1 << 20, timeout=180):
    """Streaming, resumable download with retries.

    The old version did requests.get(url).content -- buffers the whole
    response in memory with no retry/resume logic, so any dropped
    connection (common on the ~1.2GB gene2go.gz from NCBI) loses all
    progress and raises ChunkedEncodingError/IncompleteRead. This version
    streams to a .part file and resumes with a Range header on failure.
    """
    if os.path.exists(local_path):
        return local_path

    print(f"Downloading {url} ...")
    tmp_path = local_path + ".part"
    session = _make_session(max_retries)

    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(min(2 ** attempt, 30))
        resume_pos = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416:
                    # Server says nothing left to send -- already complete.
                    break
                r.raise_for_status()
                if resume_pos and r.status_code != 206:
                    # We asked for a Range but the server ignored it and sent
                    # the full file from byte 0 (common with some FTP-over-HTTPS
                    # gateways, e.g. NCBI). Appending would corrupt the file, so
                    # start over from scratch instead.
                    print(f"  server ignored Range header (status {r.status_code}); "
                          f"restarting download from byte 0")
                    resume_pos = 0
                mode = "ab" if resume_pos else "wb"
                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            break
        except requests.exceptions.RequestException as e:
            got = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            print(f"  attempt {attempt + 1}/{max_retries} failed at byte {got}: {e}")
            if attempt == max_retries - 1:
                raise
    else:
        raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")

    if tmp_path.endswith(".gz") or local_path.endswith(".gz"):
        _verify_gzip(tmp_path)

    os.replace(tmp_path, local_path)
    return local_path


def _verify_gzip(path):
    """Raise if path is not a readable gzip stream (e.g. truncated/corrupted
    download that slipped past the retry loop). Prevents caching a broken
    file that would otherwise be silently reused on every future run."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
    except Exception as e:
        os.remove(path)
        raise RuntimeError(
            f"Downloaded file {path} failed gzip integrity check ({e}); "
            f"removed corrupt file, please rerun."
        ) from e


def _ensure_decompressed(gz_path):
    """Decompress a .gz file to a sibling plain-text file (cached across runs).

    Some goatools versions/readers (e.g. Gene2GoReader in certain releases)
    don't reliably gzip-decompress the input themselves and instead read it
    as raw text, producing UnicodeDecodeError on the compressed bytes. Doing
    the decompression ourselves and handing goatools a plain file sidesteps
    that entirely.
    """
    assert gz_path.endswith(".gz")
    out_path = gz_path[:-3]
    if os.path.exists(out_path):
        return out_path
    tmp = out_path + ".part"
    with gzip.open(gz_path, "rb") as fin, open(tmp, "wb") as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            fout.write(chunk)
    os.replace(tmp, out_path)
    return out_path


def load_string_id_to_symbol(data_dir="data"):
    """Returns dict: STRING protein ID (e.g. '9606.ENSP00000...') -> gene symbol."""
    os.makedirs(data_dir, exist_ok=True)
    local_path = _download(PROTEIN_INFO_URL, os.path.join(data_dir, "9606.protein.info.v12.0.txt.gz"))
    df = pd.read_csv(local_path, sep="\t", compression="gzip")
    # columns are typically: #string_protein_id, preferred_name, protein_size, annotation
    id_col = df.columns[0]
    name_col = "preferred_name"
    return dict(zip(df[id_col], df[name_col]))


def load_go_associations(data_dir="data", go_dag_data_dir="../go_dag/data"):
    """
    Returns (godag, assoc) where assoc is dict: gene symbol -> set of GO term IDs.
    Reuses go-basic.obo from go_dag/data/ if already downloaded there, to avoid
    duplicating the same file across dataset folders.
    """
    from goatools.obo_parser import GODag
    from goatools.anno.genetogo_reader import Gene2GoReader

    os.makedirs(data_dir, exist_ok=True)

    obo_path = os.path.join(go_dag_data_dir, "go-basic.obo")
    if not os.path.exists(obo_path):
        obo_path = _download(GO_OBO_URL, os.path.join(data_dir, "go-basic.obo"))
    godag = GODag(obo_path)

    gene2go_gz_path = _download(GENE2GO_URL, os.path.join(data_dir, "gene2go.gz"),
                                 max_retries=25, timeout=120)
    gene2go_path = _ensure_decompressed(gene2go_gz_path)
    gene_info_path = _download(GENE_INFO_URL, os.path.join(data_dir, "Homo_sapiens.gene_info.gz"))

    # gene_info: GeneID -> Symbol
    with gzip.open(gene_info_path, "rt") as f:
        gene_info_df = pd.read_csv(f, sep="\t", low_memory=False)
    geneid_to_symbol = dict(zip(gene_info_df["GeneID"], gene_info_df["Symbol"]))

    reader = Gene2GoReader(gene2go_path, taxids=[int(HUMAN_TAXID)])
    ns2assoc = reader.get_ns2assc()  # {'BP': {geneid: {go_ids}}, 'CC':..., 'MF':...}

    # flatten across namespaces, convert GeneID -> Symbol
    assoc = {}
    for ns, gid2gos in ns2assoc.items():
        for gid, gos in gid2gos.items():
            symbol = geneid_to_symbol.get(gid)
            if symbol is None:
                continue
            assoc.setdefault(symbol, set()).update(gos)

    return godag, assoc


def run_go_enrichment(study_genes, population_genes, godag, assoc, alpha=0.05):
    """
    study_genes, population_genes: iterables of gene symbols.
    Returns a list of significant GOEnrichmentRecord objects, sorted by p_fdr_bh.
    """
    from goatools.go_enrichment import GOEnrichmentStudy

    study_study = GOEnrichmentStudy(
        population_genes, assoc, godag,
        propagate_counts=True, alpha=alpha, methods=["fdr_bh"]
    )
    results = study_study.run_study(study_genes)
    sig_results = [r for r in results if r.p_fdr_bh < alpha and r.enrichment == "e"]
    sig_results.sort(key=lambda r: r.p_fdr_bh)
    return sig_results


def summarize_enrichment(results, top_n=15):
    """Returns list of dicts: {go_id, name, namespace, p_fdr_bh, study_count, study_n}"""
    out = []
    for r in results[:top_n]:
        out.append({
            "go_id": r.GO,
            "name": r.name,
            "namespace": r.NS,
            "p_fdr_bh": r.p_fdr_bh,
            "study_count": r.study_count,
            "study_n": r.study_n,
        })
    return out
