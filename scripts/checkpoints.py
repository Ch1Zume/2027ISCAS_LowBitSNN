"""Pack or import model files listed in configs/checkpoints.json (stdlib only)."""
import argparse
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from common import ROOT, BUNDLES, DATASETS, checkpoint_files, sha


def pack(archive, files):
    archive = Path(archive).resolve()
    if archive.exists():
        raise FileExistsError(f'Choose a new archive: {archive}')
    for name, meta in files.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != meta['bytes'] or sha(path) != meta['sha256']:
            raise RuntimeError(f'Missing or incorrect checkpoint: {name}')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=1) as tar:
            for name in files:
                tar.add(ROOT / name, arcname=name, recursive=False)
    return len(files)


def unpack(archive, files):
    # Validate every member before installing. Only registered regular files are allowed.
    with tempfile.TemporaryDirectory(prefix='snn-checkpoints-') as directory:
        stage = Path(directory)
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in files or member.name in seen:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                meta = files[member.name]
                if member.size != meta['bytes']:
                    raise ValueError(f'Wrong checkpoint size: {member.name}')
                dest = stage / member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('xb') as out:
                    shutil.copyfileobj(src, out)
                if sha(dest) != meta['sha256']:
                    raise ValueError(f'Wrong checkpoint SHA256: {member.name}')
                seen.add(member.name)
        if not seen:
            raise ValueError('Empty checkpoint archive')
        for name in seen:
            dest = (ROOT / name).resolve()
            if not dest.is_relative_to(ROOT):
                raise ValueError(f'Checkpoint path escapes the checkout: {name}')
            if dest.exists() and sha(dest) != files[name]['sha256']:
                raise FileExistsError(f'Preserving different existing checkpoint: {name}')
        for name in seen:
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                # Stage beside the destination so the final rename is atomic across disks.
                with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as out:
                    temporary = Path(out.name)
                    with (stage / name).open('rb') as src:
                        shutil.copyfileobj(src, out)
                temporary.replace(dest)
        return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('pack', help='Create a separately uploaded model archive')
    export.add_argument('--out', type=Path, required=True)
    export.add_argument('--bundle', choices=BUNDLES)
    export.add_argument('--dataset', choices=DATASETS)
    restore = sub.add_parser('unpack', help='Verify and place uploaded files in checkpoints/')
    restore.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        count = pack(args.out, checkpoint_files(args.bundle, args.dataset))
        print(json.dumps(dict(files=count, archive=str(args.out), sha256=sha(args.out))))
    else:
        print(json.dumps(dict(imported_files=unpack(args.archive, checkpoint_files()))))


if __name__ == '__main__':
    main()
