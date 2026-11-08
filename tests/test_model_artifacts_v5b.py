"""Artifact integrity tests use tiny local fixtures, never a model or dataset."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from radar_app import model_artifacts_v5b as artifacts

ROOT=Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.folder=self.root/'runtime/models/e5-small-v5b'
        self.folder.mkdir(parents=True)
        header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        blobs={'config.json':b'{"model_type":"fixture"}',
               'tokenizer.json':b'{"vocab":{"a":1}}',
               'tokenizer_config.json':b'{"fixture":true}',
               'model.safetensors':struct.pack('<Q',len(header))+header+b'1234'}
        self.manifest={'model':'fixture/e5','revision':'revision-1','files':{}}
        for name,blob in blobs.items():
            (self.folder/name).write_bytes(blob)
            self.manifest['files'][name]={'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}
        self.write_manifest()

    def write_manifest(self):
        (self.folder/'MANIFEST.json').write_text(json.dumps(self.manifest))

    def worker(self):
        spec=importlib.util.spec_from_file_location('worker_artifact_test',ROOT/'scripts/semantic_worker_v5b.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.ROOT=self.root
        self.addCleanup(patch.stopall)
        patch.object(module.importlib.metadata,'version',return_value='test-only').start()
        return module.Worker()

    def test_tampered_config_and_tokenizer_are_rejected_even_at_the_same_size(self):
        for name in ['config.json','tokenizer.json','tokenizer_config.json']:
            with self.subTest(name=name):
                p=self.folder/name;before=p.read_bytes();p.write_bytes(b'!'+before[1:])
                with self.assertRaisesRegex(RuntimeError,'checksum'):
                    artifacts.pinned_identity('e5',root=self.root)
                p.write_bytes(before)

    def test_revision_and_file_pins_change_identity_but_json_key_order_does_not(self):
        original=artifacts.pinned_identity('e5',root=self.root)
        self.manifest=dict(reversed(list(self.manifest.items())));self.write_manifest()
        self.assertEqual(original['artifacts_fingerprint'],artifacts.pinned_identity('e5',root=self.root)['artifacts_fingerprint'])
        self.manifest['revision']='revision-2';self.write_manifest()
        self.assertNotEqual(original['artifacts_fingerprint'],artifacts.pinned_identity('e5',root=self.root)['artifacts_fingerprint'])
        self.manifest['revision']='revision-1'
        blob=b'{"vocab":{"b":2}}';(self.folder/'tokenizer.json').write_bytes(blob)
        self.manifest['files']['tokenizer.json']={'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()};self.write_manifest()
        self.assertNotEqual(original['artifacts_fingerprint'],artifacts.pinned_identity('e5',root=self.root)['artifacts_fingerprint'])

    def test_oversized_metadata_file_is_rejected_before_it_is_read(self):
        p=self.folder/'tokenizer.json'
        with p.open('wb') as f:f.truncate(artifacts.MAX_SMALL_FILE_BYTES+1)
        self.manifest['files']['tokenizer.json']['bytes']=p.stat().st_size;self.write_manifest()
        original=Path.open
        def guarded(path,*a,**kw):
            if path==p:raise AssertionError('Oversized tokenizer must not be opened')
            return original(path,*a,**kw)
        with patch.object(Path,'open',guarded):
            with self.assertRaisesRegex(RuntimeError,'limit'):
                artifacts.pinned_identity('e5',root=self.root)

    def test_manifest_path_escape_and_missing_required_file_are_rejected(self):
        before=copy.deepcopy(self.manifest)
        self.manifest['files']['../outside.json']={'bytes':0,'sha256':'0'*64};self.write_manifest()
        with self.assertRaises(RuntimeError):artifacts.pinned_identity('e5',root=self.root)
        self.manifest=before;del self.manifest['files']['tokenizer.json'];self.write_manifest()
        with self.assertRaises(RuntimeError):artifacts.pinned_identity('e5',root=self.root)

    def test_truncated_and_same_size_corrupt_weights_are_rejected_at_load(self):
        w=self.worker();w.bind('e5');p=self.folder/'model.safetensors';before=p.read_bytes()
        for blob in [before[:-1],before[:-1]+b'!']:
            with self.subTest(length=len(blob)):
                p.write_bytes(blob)
                with self.assertRaises(RuntimeError):w.load()
                self.assertIsNone(w.torch,'Integrity must fail before importing Torch')
        p.write_bytes(before)

    def test_process_descriptor_is_bound_once_and_is_not_mutable_by_the_caller(self):
        w=self.worker();w.bind('e5');first=w.describe();fingerprint=first['artifacts_fingerprint']
        first['artifact_files']['config.json']['sha256']='caller-edit'
        self.manifest['revision']='revision-2';self.write_manifest()
        w.bind('e5');second=w.describe()
        self.assertEqual(second['revision'],'revision-1')
        self.assertEqual(second['artifacts_fingerprint'],fingerprint)
        self.assertNotEqual(second['artifact_files']['config.json']['sha256'],'caller-edit')
        with self.assertRaises(ValueError):w.bind('qwen')

    def test_small_file_changed_after_bind_is_rejected_before_loading_a_model(self):
        w=self.worker();w.bind('e5');p=self.folder/'config.json';blob=p.read_bytes();p.write_bytes(b'!'+blob[1:])
        with self.assertRaisesRegex(RuntimeError,'checksum'):w.load()
        self.assertIsNone(w.torch)


if __name__=='__main__':unittest.main()
