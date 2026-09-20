#!/usr/bin/env python3
import struct, sys, numpy as np
path, out = sys.argv[1:3]
mx = np.zeros((28,1536), dtype=np.float32)
seen = np.zeros(28, dtype=np.int64)
with open(path,'rb') as f:
    assert f.read(8) == b'LLDUMP1\0'
    while True:
        b=f.read(2)
        if not b: break
        (nl,)=struct.unpack('<H',b)
        name=f.read(nl).decode()
        (nv,)=struct.unpack('<Q',f.read(8))
        raw=f.read(nv*4)
        if name.startswith('ffn_norm-'):
            layer=int(name.split('-')[-1])
            a=np.frombuffer(raw,dtype='<f4').reshape(-1,1536)
            mx[layer]=np.maximum(mx[layer],np.max(np.abs(a),axis=0))
            seen[layer]+=a.shape[0]
np.savez(out, activation_max=mx, tokens=seen)
print('tokens/layer', seen.tolist())
print('range', float(mx.min()), float(mx.max()))
