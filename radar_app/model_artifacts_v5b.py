"""Bounded stdlib checks for pinned model artifacts; never imports inference code."""
import copy
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import struct

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES={'e5':'e5-small-v5b','qwen':'qwen-embedding-v5b','nli':'nli-minilm-v5b'}
MAX_MANIFEST_BYTES=256*1024
MAX_SMALL_FILE_BYTES=32*1024**2
MAX_TOTAL_SMALL_BYTES=64*1024**2
MAX_WEIGHT_BYTES=4*1024**3
MAX_HEADER_BYTES=16*1024**2
MAX_FILES=64
REQUIRED={'config.json','tokenizer.json','tokenizer_config.json','model.safetensors'}


def _path(directory,name):
    rel=PurePosixPath(name)
    if (not name or len(name)>256 or rel.is_absolute() or '\\' in name or '\x00' in name
            or any(p in {'.','..'} for p in rel.parts)):
        raise RuntimeError('Invalid artifact path')
    path=directory/name
    if not path.resolve().is_relative_to(directory.resolve()):raise RuntimeError('Artifact path escapes model directory')
    return path


def _bounded_read(path,limit):
    if not path.is_file() or path.stat().st_size>limit:raise RuntimeError('Artifact size limit or missing file: '+path.name)
    with path.open('rb') as f:data=f.read(limit+1)
    if len(data)>limit:raise RuntimeError('Artifact size limit: '+path.name)
    return data


def _validate_manifest(manifest):
    if not isinstance(manifest,dict):raise RuntimeError('Invalid model manifest')
    for field in ['model','revision']:
        value=manifest.get(field)
        if not isinstance(value,str) or not 1<=len(value)<=256:raise RuntimeError('Invalid manifest '+field)
    files=manifest.get('files')
    if not isinstance(files,dict) or not REQUIRED<=files.keys() or len(files)>MAX_FILES:
        raise RuntimeError('Manifest file limit or missing required artifacts')
    total=0
    for name,meta in files.items():
        if not isinstance(name,str) or not isinstance(meta,dict):raise RuntimeError('Invalid artifact entry')
        size=meta.get('bytes');digest=meta.get('sha256')
        if isinstance(size,bool) or not isinstance(size,int) or size<0:raise RuntimeError('Invalid artifact bytes')
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest):raise RuntimeError('Invalid artifact checksum')
        if name=='model.safetensors':
            if not 8<size<=MAX_WEIGHT_BYTES:raise RuntimeError('Weight size limit')
        else:
            if size>MAX_SMALL_FILE_BYTES:raise RuntimeError('Small artifact size limit')
            total+=size
    if total>MAX_TOTAL_SMALL_BYTES:raise RuntimeError('Total small artifact size limit')


def _verified_small(directory,name,meta):
    path=_path(directory,name)
    if not path.is_file() or path.stat().st_size!=meta['bytes']:raise RuntimeError('Artifact byte size differs: '+name)
    data=_bounded_read(path,MAX_SMALL_FILE_BYTES)
    if len(data)!=meta['bytes']:raise RuntimeError('Artifact byte size differs: '+name)
    if hashlib.sha256(data).hexdigest()!=meta['sha256']:raise RuntimeError('Artifact checksum differs: '+name)
    return data


def verify_weights(path,expected_sha256,expected_bytes=None):
    """Check bounded safetensors structure and stream the SHA before model loading."""
    path=Path(path);size=path.stat().st_size
    if not 8<size<=MAX_WEIGHT_BYTES:raise RuntimeError('Weight size limit')
    if expected_bytes is not None and size!=expected_bytes:raise RuntimeError('Pinned weight byte size differs')
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise RuntimeError('Invalid model header size')
        header_size=struct.unpack('<Q',raw)[0]
        if not 0<header_size<=min(MAX_HEADER_BYTES,size-8):raise RuntimeError('Invalid model header size limit')
        header=json.loads(f.read(header_size));ends=[]
        if not isinstance(header,dict):raise RuntimeError('Invalid model header')
        for name,meta in header.items():
            if name=='__metadata__':continue
            offsets=meta.get('data_offsets') if isinstance(meta,dict) else None
            if (not isinstance(offsets,list) or len(offsets)!=2
                    or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in offsets)
                    or offsets[1]<offsets[0]):raise RuntimeError('Invalid tensor offsets')
            ends.append(offsets[1])
        if not ends or 8+header_size+max(ends)!=size:raise RuntimeError('Incomplete model file size')
        f.seek(0);digest=hashlib.sha256();read=0
        while chunk:=f.read(1024*1024):
            read+=len(chunk)
            if read>size:raise RuntimeError('Weight size changed during checksum')
            digest.update(chunk)
        if read!=size or digest.hexdigest()!=expected_sha256:raise RuntimeError('Pinned model checksum differs')


def validate_artifacts(directory,manifest,*,verify_weight_sha256=False):
    """Validate against a captured manifest, without refreshing its identity."""
    directory=Path(directory);_validate_manifest(manifest);config=None
    for name,meta in manifest['files'].items():
        path=_path(directory,name)
        if name=='model.safetensors':
            if not path.is_file() or path.stat().st_size!=meta['bytes']:raise RuntimeError('Pinned weight byte size differs')
            if verify_weight_sha256:verify_weights(path,meta['sha256'],meta['bytes'])
        else:
            data=_verified_small(directory,name,meta)
            if name=='config.json':config=json.loads(data)
    if not isinstance(config,dict):raise RuntimeError('Invalid model config')
    # These optional files affect tokenization when present, so cannot be unpinned.
    for name in ['added_tokens.json','special_tokens_map.json','vocab.json','vocab.txt','merges.txt','tokenizer.model']:
        if (directory/name).exists() and name not in manifest['files']:raise RuntimeError('Unpinned tokenizer artifact: '+name)
    return config


def pinned_identity(backend,root=ROOT):
    """Validate all declared small files and return canonical, revision-bound pins.

    Weight SHA is an expected pin here; its bytes are hashed by the worker at load.
    """
    if backend not in DIRECTORIES:raise ValueError('Unknown pinned local backend')
    directory=Path(root)/'runtime/models'/DIRECTORIES[backend]
    manifest=json.loads(_bounded_read(directory/'MANIFEST.json',MAX_MANIFEST_BYTES))
    _validate_manifest(manifest)
    config=validate_artifacts(directory,manifest)
    canonical=json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
    manifest_sha=hashlib.sha256(canonical).hexdigest()
    fingerprint=hashlib.sha256(json.dumps({'schema':1,'backend':backend,'manifest_sha256':manifest_sha},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(backend=backend,model=manifest['model'],revision=manifest['revision'],
                weights_sha256=manifest['files']['model.safetensors']['sha256'],
                manifest_sha256=manifest_sha,artifacts_fingerprint=fingerprint,
                files=copy.deepcopy(manifest['files']),manifest=copy.deepcopy(manifest),config=config)
