"""Package only the two registered map files, manifests and original Kaggle replay. No scoring."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_STORED

import numpy as np
from compare_axes import ORIGINAL_MAPS, ORIGINAL_REPLAY, VERSIONS, frozen_math, require, sha


OLD_INPUT_ARCHIVE = 'eee02923ce3e2aec28f62e66e8d5580097bd65c22b2584721f2e1b03b68b9517'
OLD_RETURN_ARCHIVE = '093a4d1901fe392e9181897287b86cddf686bd46d1a063a706c733cf7ca58fda'
CORRECTED_MAPS = '0d4abbe1ac37e4d6974601c40ad65fd3bcaf60b81b4bdca63878925657a32150'
CORRECTED_MANIFEST = '7c260c2490f49ecf2eba7bfdf12610a17c8862d6ac20e853a3c8980617c12270'


def pack(a):
    require(sha(a.legacy_input.read_bytes())==OLD_INPUT_ARCHIVE and sha(a.legacy_return.read_bytes())==OLD_RETURN_ARCHIVE,'Unrecognized original archive')
    with ZipFile(a.legacy_input) as old, ZipFile(a.legacy_return) as returned:
        require(old.testzip() is None and returned.testzip() is None,'Original archives CRC')
        blobs={'legacy_raster/'+n:old.read(n) for n in ('maps.npz','manifest.json')}
        blobs['legacy_replay.npz']=returned.read('causal_geometry_replay.npz')
    blobs.update({'source_label/'+n:(a.corrected/n).read_bytes() for n in ('maps.npz','manifest.json')})
    for k,expected in [('legacy_raster/maps.npz',ORIGINAL_MAPS),('legacy_replay.npz',ORIGINAL_REPLAY),
                       ('source_label/maps.npz',CORRECTED_MAPS),('source_label/manifest.json',CORRECTED_MANIFEST)]:
        require(sha(blobs[k])==expected, 'Frozen member '+k)
    funcs=frozen_math(a.evaluator.read_bytes()); hashes={}
    for version in VERSIONS:
        with np.load(io.BytesIO(blobs[version+'/maps.npz']),allow_pickle=False) as data:
            hashes[version]=funcs.array_digest(data)
    sources=json.loads(blobs['legacy_raster/manifest.json'])['sources']
    require(sources==json.loads(blobs['source_label/manifest.json'])['sources'],'Reference source changes')
    for clip,source in sources.items():
        folder=clip.rsplit('_',1)[0]
        require(sha((a.metadata/'labels'/(clip+'.txt')).read_bytes())==source['label_sha256'],'Original labels')
        require(sha((a.metadata/'videos'/folder/'calib.json').read_bytes())==source['calibration_sha256'],'Original calibration')
    with ZipFile(a.output,'w',compression=ZIP_STORED) as archive:
        for name,raw in sorted(blobs.items()): archive.writestr(ZipInfo(name),raw)
    lock={'private_input_sha256':sha(a.output.read_bytes()), 'member_sha256':{k:sha(v) for k,v in sorted(blobs.items())},
          'original_input_archive_sha256':OLD_INPUT_ARCHIVE,'original_return_archive_sha256':OLD_RETURN_ARCHIVE,
          'map_array_sha256':hashes,'title':'Historical map convention and source-label axis correction; original archive remains immutable.'}
    a.lock.write_text(json.dumps(lock,indent=2)+'\n')
    print('Private working input',a.output.stat().st_size,'bytes;',lock['private_input_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('legacy-input','legacy-return','corrected','metadata','evaluator','output','lock'):
        p.add_argument('--'+name,type=Path,required=True)
    pack(p.parse_args())
