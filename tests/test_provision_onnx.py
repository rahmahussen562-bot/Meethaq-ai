"""Offline verification of model setup integrity, limits, and safe publication."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import provision_onnx as setup


def archive_bytes(extra=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in sorted(setup.MODEL_FILES):
            content = ("fixture " + name).encode()
            member = tarfile.TarInfo("onnx/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if extra is not None:
            archive.addfile(extra, io.BytesIO(b"x") if extra.isfile() else None)
    return buffer.getvalue()


class Response(io.BytesIO):
    def __init__(self, body, length=None):
        super().__init__(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def geturl(self):
        return setup.MODEL_DOWNLOAD_URL


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.project = Path(self.directory.name)
        self.project_patch = patch.object(setup, "PROJECT_DIR", self.project)
        self.project_patch.start()
        self.root = setup.cache_root()

    def tearDown(self):
        self.project_patch.stop()
        self.directory.cleanup()

    def fixture(self, payload=None):
        payload = archive_bytes() if payload is None else payload
        archive = self.project / "fixture.tar.gz"
        archive.write_bytes(payload)
        return archive, hashlib.sha256(payload).hexdigest()

    def test_missing_cache_never_downloads_without_explicit_flag(self):
        with patch.object(setup, "urlopen") as network, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(setup.main([]), 1)
        network.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_checksum_is_verified_before_archive_parsing_or_cache_writes(self):
        archive, _ = self.fixture()
        with patch.object(setup.tarfile, "open") as parse:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                setup.install_archive(archive, self.root)
        parse.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_local_import_publishes_complete_verified_cache_and_reuses_it(self):
        archive, checksum = self.fixture()
        with patch.object(setup, "MODEL_SHA256", checksum), patch.object(setup, "urlopen") as network:
            model_dir = setup.install_archive(archive, self.root)
            self.assertEqual(model_dir, (self.root / "onnx").resolve())
            manifest = json.loads((model_dir / "provisioned.json").read_text())
            self.assertEqual(set(manifest["files"]), setup.MODEL_FILES)
            self.assertEqual(setup.install_archive(archive, self.root), model_dir)
            self.assertEqual(list(self.root.glob("extract-*")), [])
        network.assert_not_called()

    def test_traversal_links_and_unexpected_files_are_rejected_before_writes(self):
        for name, kind in [("onnx/../../escaped", tarfile.REGTYPE),
                           ("onnx\\model.onnx", tarfile.REGTYPE),
                           ("/onnx/model.onnx", tarfile.REGTYPE),
                           ("onnx/link", tarfile.SYMTYPE),
                           ("onnx/unknown.json", tarfile.REGTYPE)]:
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.size = 1 if kind == tarfile.REGTYPE else 0
                member.linkname = "../../escaped"
                archive, checksum = self.fixture(archive_bytes(member))
                with patch.object(setup, "MODEL_SHA256", checksum):
                    with self.assertRaises(ValueError):
                        setup.install_archive(archive, self.root)
                self.assertFalse(self.root.exists())
                self.assertFalse((self.project / "escaped").exists())

    def test_duplicate_files_and_extracted_size_limit_are_rejected(self):
        duplicate = tarfile.TarInfo("onnx/model.onnx")
        duplicate.size = 1
        archive, checksum = self.fixture(archive_bytes(duplicate))
        with patch.object(setup, "MODEL_SHA256", checksum):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                setup.install_archive(archive, self.root)
        archive, checksum = self.fixture()
        with patch.object(setup, "MODEL_SHA256", checksum), patch.object(setup, "MAX_EXTRACTED_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "size limit"):
                setup.install_archive(archive, self.root)
        self.assertFalse(self.root.exists())

    def test_disk_space_failure_never_creates_partial_model(self):
        archive, checksum = self.fixture()
        with patch.object(setup, "MODEL_SHA256", checksum), patch.object(setup.shutil, "disk_usage", return_value=SimpleNamespace(free=setup.DISK_RESERVE_BYTES)):
            with self.assertRaisesRegex(OSError, "disk space"):
                setup.install_archive(archive, self.root)
        self.assertFalse(self.root.exists())
        with patch.object(setup, "MODEL_SHA256", checksum), patch.object(setup, "require_space", side_effect=[None, None, OSError("disk filled during extraction")]):
            with self.assertRaisesRegex(OSError, "during extraction"):
                setup.install_archive(archive, self.root)
        self.assertFalse((self.root / "onnx").exists())
        self.assertEqual(list(self.root.glob("extract-*")), [])

    def test_explicit_download_installs_model_and_removes_archive(self):
        body = archive_bytes()
        checksum = hashlib.sha256(body).hexdigest()
        with patch.object(setup, "MODEL_SHA256", checksum), patch.object(setup, "MAX_ARCHIVE_BYTES", 1024), patch.object(setup, "MAX_EXTRACTED_BYTES", 1024), patch.object(setup, "require_space", wraps=setup.require_space) as space, patch.object(setup, "urlopen", return_value=Response(body, len(body))) as network:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(setup.main(["--download"]), 0)
        network.assert_called_once_with(setup.MODEL_DOWNLOAD_URL, timeout=setup.DOWNLOAD_TIMEOUT_SECONDS)
        self.assertEqual(space.call_args_list[1].args, (self.root, len(body)))
        self.assertEqual(output.getvalue().strip(), str((self.root / "onnx").resolve()))
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_truncated_oversized_and_bad_checksum_downloads_remove_partial_files(self):
        for body, length, limit, message in [(b"x", 2, 100, "advertised"),
                                             (b"xx", None, 1, "size limit"),
                                             (b"x", 1, 100, "SHA-256")]:
            with self.subTest(message=message):
                with patch.object(setup, "MAX_ARCHIVE_BYTES", limit), patch.object(setup, "MAX_EXTRACTED_BYTES", 1024), patch.object(setup, "urlopen", return_value=Response(body, length)):
                    with self.assertRaisesRegex(ValueError, message):
                        setup.download_archive(self.root)
                self.assertEqual(list(self.root.glob("*.part")), [])
                self.assertFalse((self.root / "onnx").exists())
        with patch.object(setup, "require_space", side_effect=[None, None, OSError("disk filled during download")]), patch.object(setup, "urlopen", return_value=Response(b"x", 1)):
            with self.assertRaisesRegex(OSError, "during download"):
                setup.download_archive(self.root)
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_changed_cached_file_is_rejected_without_network_access(self):
        archive, checksum = self.fixture()
        with patch.object(setup, "MODEL_SHA256", checksum):
            model_dir = setup.install_archive(archive, self.root)
            (model_dir / "model.onnx").write_text("changed")
            with patch.object(setup, "urlopen") as network, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(setup.main(["--download"]), 1)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
