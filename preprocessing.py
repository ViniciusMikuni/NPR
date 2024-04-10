import numpy as np
import pylhe
from lhereader import LHEReader
import h5py as h5

weights1 = []
weights2 = []

is_weight1 = True #Choose set of PDF weights to save 
with open('/global/cfs/cdirs/m3929/NPR/ttbb-events-oldPDF-phase1-2-3-4.lhe') as f:
    for line in f:
        if 'pdf1' in line:        
            weight = line.split(' ')[4]
            if weight == '': #negative weights
                weight = line.split(' ')[3]
            if weight == '':
                weight = line.split(' ')[2] #negative weights smaller than 1
            if is_weight1:
                weights1.append(weight)
                is_weight1=False
            else:
                weights2.append(weight)
                is_weight1=True

weights1 = np.array(weights1).astype(np.float32)
weights2 = np.array(weights2).astype(np.float32)


data = []
reader = LHEReader('/global/cfs/cdirs/m3929/NPR/ttbb-events-oldPDF-phase1-2-3-4.lhe')
for iev, event in enumerate(reader):
    evt = []
    for particle in event.particles:
        if particle.status==1:
            evt.append([particle.px,particle.py,particle.pz,particle.pdgid])
    if np.array(evt).shape != (5,4): #first additional emission missing
        evt.append([0.0,0.0,0.0,0.0])        
    data.append(evt)
    
        
data = np.array(data).reshape([-1,5,4])
print(data.shape)
print(weights1.shape)

with h5.File('/global/cfs/cdirs/m3929/NPR/ttbb.h5', "w") as fh5:
    dset = fh5.create_dataset('data', data=data)
    dset = fh5.create_dataset('weights1', data=weights1)
    dset = fh5.create_dataset('weights2', data=weights2)
