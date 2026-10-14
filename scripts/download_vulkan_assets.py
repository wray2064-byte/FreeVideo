"""Fetch pinned, integrity-checked assets for the experimental AMD test."""
import concurrent.futures, hashlib, json, pathlib, subprocess, sys, urllib.request

repo = pathlib.Path(__file__).resolve().parents[1]
root = repo / 'experimental-models'
deps = json.loads((repo / 'freevideo_engine/dependencies.json').read_text())['models']
edge = 'https://huggingface.co/OpenVDN/vdn-minimax-h3-edge/resolve/' + deps['edge_modelscope']['hf_snapshot_revision'] + '/cache/'
manifest_path = root / 'cache/manifest.json'
manifest_sha = 'ad339c92679300fd43f57a7443d97b7392cce9910237c414ce32976509b18bf7'
if not manifest_path.exists():
    with urllib.request.urlopen(edge + 'manifest.json', timeout=90) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != manifest_sha:
        raise ValueError('Pinned FP8 manifest integrity failure')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(data)
data = manifest_path.read_bytes()
if hashlib.sha256(data).hexdigest() != manifest_sha:
    raise ValueError('Pinned FP8 manifest integrity failure')
manifest = json.loads(data)
jobs = []
for row in manifest['groups']:
    jobs.append((edge + row['file'], root / 'cache' / row['file'], row['bytes'], row['sha256']))
for table in manifest['adaln_tables']:
    if table['producer']['task'] == 't2va':
        for row in table['files']:
            jobs.append((edge + row['file'], root / 'cache' / row['file'], row['bytes'], row['sha256']))
for row in json.loads((repo / 'freevideo_engine/model_files.json').read_text()):
    name = row['file']
    if not name.startswith('h3-base/') and name != 'stage-dmd-step-250/model_spec.json':
        continue
    if '/transformer/diffusion_pytorch_model' in name:
        continue
    jobs.append((f"https://huggingface.co/{row['repo']}/resolve/{row['revision']}/{name}", root / name, row['bytes'], row.get('download_sha256') or row.get('sha256')))
encoder = root / 'encoder' / pathlib.Path(deps['encoder_file']).name
jobs.append((f"https://huggingface.co/{deps['encoder_repo']}/resolve/{deps['encoder_revision']}/{deps['encoder_file']}", encoder, None, deps['encoder_sha256']))
jobs.sort(key=lambda job: 0 if job[1] == encoder else 1)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()

def fetch(job):
    url, path, size, sha = job
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and (size is None or path.stat().st_size == size) and (not sha or digest(path) == sha)):
        if size is not None and size < 8 << 20:
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            if len(data) != size or (sha and hashlib.sha256(data).hexdigest() != sha):
                raise ValueError('Integrity failure: ' + str(path))
            path.write_bytes(data)
        else:
            subprocess.run([sys.executable, str(repo / 'scripts/range_download.py'), url, str(path)], check=True)
    if size is not None and path.stat().st_size != size or sha and digest(path) != sha:
        raise ValueError('Integrity failure: ' + str(path))
    print('Verified ' + str(path.relative_to(root)), flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(fetch, jobs))
print('All experimental model assets verified.', flush=True)
