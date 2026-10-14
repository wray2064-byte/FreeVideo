"""Resumable ranges for immutable public model URLs; caller verifies full SHA-256."""
import concurrent.futures, json, pathlib, sys, time, urllib.request

url, destination = sys.argv[1:3]
destination = pathlib.Path(destination)
for attempt in range(8):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=45) as response:
            size, resolved = int(response.headers['Content-Length']), response.url
        break
    except Exception:
        if attempt == 7:
            raise
        time.sleep(5)
partial = destination.with_suffix(destination.suffix + '.partial')
marker = destination.with_suffix(destination.suffix + '.ranges.json')
chunk = 32 << 20
identity = dict(url=url, bytes=size, chunk_bytes=chunk)
done = set()
if partial.exists() and partial.stat().st_size == size and marker.exists():
    saved = json.loads(marker.read_text())
    if saved.get('identity') == identity:
        done = set(saved['completed'])
if not done:
    with partial.open('wb') as stream:
        stream.truncate(size)

def fetch(start):
    end = min(size - 1, start + chunk - 1)
    for attempt in range(8):
        try:
            request = urllib.request.Request(resolved, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                    raise ValueError('Server did not honor range')
                data = response.read()
            if len(data) != end - start + 1:
                raise ValueError('Truncated range')
            with partial.open('r+b') as stream:
                stream.seek(start)
                stream.write(data)
            return start
        except Exception:
            if attempt == 7:
                raise
            time.sleep(5)

starts = list(range(0, size, chunk))
last = -1
errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(fetch, start) for start in starts if start not in done]
    for future in concurrent.futures.as_completed(futures):
        try:
            done.add(future.result())
        except Exception as error:
            errors.append(error)
            continue
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(identity=identity, completed=sorted(done))))
        temporary.replace(marker)
        percent = len(done) * 100 // len(starts)
        if percent // 10 != last:
            print(f'{destination.name}: {percent}%', flush=True)
            last = percent // 10
if errors:
    raise errors[0]
if done != set(starts):
    raise ValueError('Missing model ranges')
partial.replace(destination)
marker.unlink(missing_ok=True)
print(f'Completed {destination.name}: {size} bytes', flush=True)
