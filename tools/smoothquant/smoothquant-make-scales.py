#!/usr/bin/env python3
import sys, numpy as np
from gguf import GGUFReader
model, stats, outdir = sys.argv[1:4]
a=np.load(stats)['activation_max']
r=GGUFReader(model,'r')
t={x.name:x for x in r.tensors}
wmax=np.zeros_like(a)
for i in range(28):
 g=np.asarray(t[f'blk.{i}.ffn_gate.weight'].data,dtype=np.float32)
 u=np.asarray(t[f'blk.{i}.ffn_up.weight'].data,dtype=np.float32)
 wmax[i]=np.maximum(np.max(np.abs(g),axis=0),np.max(np.abs(u),axis=0))
for alpha in (0.4,0.5,0.6):
 s=np.power(np.maximum(a,1e-6),alpha)/np.power(np.maximum(wmax,1e-6),1-alpha)
 s=np.clip(s,1e-3,1e3).astype('<f4')
 path=f'{outdir}/sq-scales-a{int(alpha*10):02d}.bin'
 s.tofile(path)
 print(path, float(s.min()), float(s.max()), float(np.median(s)))
