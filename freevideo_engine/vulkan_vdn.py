"""Pinned VDN import overlay with explicit synchronous host/Vulkan execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def activate():
    from .paths import add_vdn, data_root
    from .macos_vdn import adapt
    source = add_vdn()
    expected = json.loads(Path(__file__).with_name('dependencies.json').read_text())['vdn']['commit']
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True)
    if actual != expected or dirty.strip():
        raise ValueError('Vulkan overlay requires the clean pinned VDN source')
    names = subprocess.check_output(['git', 'ls-files', '-z', 'src'], cwd=source).decode().split('\0')
    contents = {}
    for name in filter(None, names):
        path = source / name
        if path.is_symlink():
            raise ValueError('Unexpected source symlink')
        data = path.read_bytes()
        if name.endswith('.py'):
            text = data.decode('utf-8').replace('\r\n', '\n')
            text = adapt(name, text)
            if name == 'src/inference/render.py':
                text = text.replace('torch.mps.synchronize()', 'None')
                text = text.replace('torch.autocast(device_type="cuda", dtype=torch.float16)',
                                    'torch.autocast(device_type="cpu", enabled=False)')
            compile(text, name, 'exec')
            data = text.encode()
        contents[name] = data
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    identity = {'upstream_commit': actual, 'policy': 'host-backed-vulkan-v1', 'files': hashes}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    target = data_root() / 'vendor' / ('vdn-vulkan-' + key)
    target.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError('Vulkan overlay identity mismatch: ' + name)
        if not path.exists():
            path.write_bytes(data)
    (target / 'identity.json').write_text(json.dumps(identity, indent=2))
    if 'src' in sys.modules and not Path(sys.modules['src'].__file__).resolve().is_relative_to(target):
        raise RuntimeError('VDN was already imported; launch a fresh Vulkan worker')
    sys.path.insert(0, str(target))
    return target
