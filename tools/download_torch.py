"""Resume an interrupted official PyTorch wheel using bounded parallel ranges.

Only the expected PyTorch HTTPS host is used; verify official index SHA256 before
the wheel is used. This is an installation helper, not part of the video pipeline.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen


INDEX = 'https://download.pytorch.org/whl/cu126/torch/'
NAME = 'torch-2.8.0+cu126-cp311-cp311-win_amd64.whl'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(v for k, v in attrs if k == 'href')


def main():
    prefix = Path(sys.argv[1])
    root = Path('artifacts/downloads')
    root.mkdir(parents=True, exist_ok=True)
    parser = Links()
    with urlopen(INDEX, timeout=60) as response:
        parser.feed(response.read().decode())
    link = next(link for link in parser.links if unquote(urlsplit(link).path).endswith('/' + NAME))
    original = urlsplit(urljoin(INDEX, link))
    expected = original.fragment.removeprefix('sha256=')
    assert len(expected) == 64, 'Official SHA256 missing'
    url = 'https://download.pytorch.org' + original.path
    with urlopen(Request(url, method='HEAD'), timeout=60) as response:
        total = int(response.headers['Content-Length'])
    offset = prefix.stat().st_size
    assert 0 < offset < total
    chunk_size = (total - offset + 11) // 12

    def fetch(task):
        i, begin, end = task
        target = root / f'torch-{i}.part'
        required = end - begin + 1
        for attempt in range(4):
            have = target.stat().st_size if target.exists() else 0
            if have == required:
                return target
            assert have < required
            req = Request(url, headers={'Range': f'bytes={begin + have}-{end}'})
            try:
                with urlopen(req, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'].startswith(f'bytes {begin + have}-{end}/')
                    with target.open('ab') as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                assert target.stat().st_size == required
                print(f'Part {i + 1}/12 complete', flush=True)
                return target
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)

    tasks = [(i, start, min(total - 1, start + chunk_size - 1))
             for i, start in enumerate(range(offset, total, chunk_size))]
    print(f'Resuming {offset}/{total} bytes, {len(tasks)} connections', flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        parts = list(pool.map(fetch, tasks))
    output = root / NAME
    digest = hashlib.sha256()
    with output.open('wb') as dest:
        for part in [prefix, *parts]:
            with part.open('rb') as src:
                while block := src.read(1024 * 1024):
                    digest.update(block)
                    dest.write(block)
    if digest.hexdigest() != expected:
        raise RuntimeError('SHA256 mismatch: do not install this wheel')
    print(f'Verified SHA256 {expected}: {output}', flush=True)


if __name__ == '__main__':
    main()
