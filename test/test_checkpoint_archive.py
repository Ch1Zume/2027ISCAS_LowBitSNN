"""Uploaded model archives must be verified before any checkpoint is installed."""
import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import checkpoints


def metadata(payload):
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def archive(path, members):
    with tarfile.open(path, 'w:gz') as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def test_separate_upload_round_trip_and_idempotent_import(tmp_path, monkeypatch):
    source, target = tmp_path / 'source', tmp_path / 'new-clone'
    name, payload = 'checkpoints/lut_final/NMNIST.pth', b'checkpoint-bytes'
    path = source / name
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    manifest = {name: metadata(payload)}
    monkeypatch.setattr(checkpoints, 'ROOT', source)
    packed = tmp_path / 'models.tar.gz'
    assert checkpoints.pack(packed, manifest) == 1
    monkeypatch.setattr(checkpoints, 'ROOT', target)
    assert checkpoints.unpack(packed, manifest) == 1
    assert (target / name).read_bytes() == payload
    assert checkpoints.unpack(packed, manifest) == 1


@pytest.mark.parametrize('bad_name,bad_payload', [
    ('../escape.pth', b'bad'),
    ('checkpoints/b.pth', b'corrupt'),
    ('checkpoints/a.pth', b'good'),  # duplicate member
])
def test_rejects_invalid_archive_before_installing_valid_prefix(tmp_path, monkeypatch, bad_name, bad_payload):
    target = tmp_path / 'new-clone'
    monkeypatch.setattr(checkpoints, 'ROOT', target)
    packed = tmp_path / 'models.tar.gz'
    archive(packed, [('checkpoints/a.pth', b'good'), (bad_name, bad_payload)])
    files = {'checkpoints/a.pth': metadata(b'good'), 'checkpoints/b.pth': metadata(b'correct')}
    with pytest.raises(ValueError):
        checkpoints.unpack(packed, files)
    assert not target.exists()
    assert not (tmp_path / 'escape.pth').exists()


def test_import_preserves_different_existing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, 'ROOT', tmp_path)
    path = tmp_path / 'checkpoints/a.pth'
    path.parent.mkdir()
    path.write_bytes(b'local-new-training')
    packed = tmp_path / 'models.tar.gz'
    archive(packed, [('checkpoints/a.pth', b'frozen-model')])
    with pytest.raises(FileExistsError):
        checkpoints.unpack(packed, {'checkpoints/a.pth': metadata(b'frozen-model')})
    assert path.read_bytes() == b'local-new-training'
