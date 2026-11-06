"""Verified local assets. This module neither downloads nor loads models on import."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen


class AssetError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class AssetStore:
    def __init__(self, root='artifacts', manifest=None):
        self.root = Path(root).expanduser().resolve()
        if manifest is None:
            checkout = Path.cwd() / 'assets' / 'asset_manifest.json'
            bundled = Path(__file__).with_name('resources') / 'asset_manifest.json'
            manifest = checkout if checkout.is_file() else bundled
        manifest = Path(manifest)
        if not manifest.is_file():
            raise AssetError('Asset manifest not found. Run from the checkout or pass --manifest PATH.')
        self.manifest_path = manifest.resolve()
        content = json.loads(manifest.read_text())
        entries = content['assets']
        if isinstance(entries, dict):
            entries = [dict(item, id=key) for key, item in entries.items()]
        self.entries = {item['id']: item for item in entries}
        if len(self.entries) != len(entries):
            raise AssetError('Duplicate asset IDs in manifest.')

    def entry(self, asset_id):
        try:
            return self.entries[asset_id]
        except KeyError:
            raise AssetError('Unknown asset: ' + asset_id + '. Use swaap assets list.') from None

    def destination(self, asset_id):
        entry = self.entry(asset_id)
        filename = entry.get('filename') or asset_id
        candidate = (self.root / filename).resolve()
        if self.root not in candidate.parents or candidate == self.root:
            raise AssetError('Unsafe asset filename in manifest.')
        return candidate

    def verify(self, asset_id, path=None):
        entry = self.entry(asset_id)
        path = Path(path) if path else self.destination(asset_id)
        if not path.is_file():
            raise AssetError('Missing asset ' + asset_id + '. Use swaap assets download, or assets import --file PATH when no public URL is available.')
        expected_hash = entry.get('sha256')
        expected_size = entry.get('bytes', entry.get('size_bytes'))
        if not expected_hash or expected_size is None:
            raise AssetError('Exact identity is unavailable for ' + asset_id + '; it cannot be certified as a paper asset.')
        actual_size = path.stat().st_size
        if actual_size != int(expected_size):
            raise AssetError('Size mismatch for ' + asset_id)
        digest = sha256(path)
        if digest != expected_hash:
            raise AssetError('SHA-256 mismatch for ' + asset_id)
        return dict(id=asset_id, path=str(path.resolve()), bytes=actual_size, sha256=digest, status='VERIFIED')

    def import_file(self, asset_id, source):
        verified = self.verify(asset_id, source)
        destination = self.destination(asset_id)
        if destination.exists():
            return self.verify(asset_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.import-', dir=destination.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            self.verify(asset_id, temporary)
            # A hard link publishes create-only, so concurrent imports cannot overwrite.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self.verify(asset_id)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return self.verify(asset_id)

    def download(self, asset_id):
        entry = self.entry(asset_id)
        destination = self.destination(asset_id)
        if destination.exists():
            return self.verify(asset_id)
        urls = entry.get('urls') or []
        if not urls:
            raise AssetError('No verified public download for ' + asset_id + '. Hosting or permission is pending; use checksum-verified local import if you have lawful access. See assets/README.md.')
        if not entry.get('sha256') or entry.get('bytes', entry.get('size_bytes')) is None:
            raise AssetError('Refusing a download without an exact checksum and size.')
        url = urls[0]
        if isinstance(url, dict):
            url = url.get('url')
        if not isinstance(url, str) or urlparse(url).scheme != 'https':
            raise AssetError('Manifest downloads must use an explicitly recorded HTTPS URL.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
        try:
            with os.fdopen(fd, 'wb') as out, urlopen(url, timeout=60) as stream:
                expected = int(entry.get('bytes', entry.get('size_bytes')))
                written = 0
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > expected:
                        raise AssetError('Download exceeds the pinned size; preserved destination unchanged.')
                    out.write(block)
            self.verify(asset_id, temporary)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self.verify(asset_id)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return self.verify(asset_id)
