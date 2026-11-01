"""MicroPython/CPython inference for the existing 161-parameter ContourGate.

No numpy, PC, cloud, pretrained backbone, or external runtime is required.
The output is a filter update gain, NOT an object class or a calibrated probability.
"""
import math
from contour_gate_weights import W1, B1, W2, B2, MODEL_ID

def clip(value,lo,hi):
    return lo if value<lo else hi if value>hi else value

def make_features(obs,pred,previous,velocity,dt,quality,height):
    delta=obs-pred
    adjacent=obs-previous
    return (clip(delta/.15,-5,5),clip(adjacent/.15,-5,5),
            clip(velocity*dt/.15,-3,3),min(abs(delta)/.15,5),
            clip(quality,0,1),min(height/80,2),clip(dt/.05,.2,6),
            min(abs(adjacent)/.15,5))

def predict(x):
    total=B2[0]
    for j in range(16):
        acc=B1[j]
        for i in range(8):
            acc+=x[i]*W1[i*16+j]
        total+=math.tanh(acc)*W2[j]
    return clip(1/(1+math.exp(-clip(total,-30,30))),.03,.95)

class TemporalDepthFilter:
    def __init__(self):self.reset()

    def reset(self):
        self.value=None
        self.previous_obs=None
        self.velocity=0.0
        self.last_gate=1.0

    def update(self,depth,dt,quality=1.0,height_px=30.0):
        if not math.isfinite(depth) or depth<=0:
            raise ValueError('Depth must be finite and positive')
        obs=math.log(depth);dt=clip(dt,.01,.3)
        if self.value is None:
            self.value=obs;self.previous_obs=obs
            return depth,1.0
        pred=self.value+self.velocity*dt
        gate=predict(make_features(obs,pred,self.previous_obs,self.velocity,dt,quality,height_px))
        delta=obs-pred
        self.value=pred+gate*delta
        self.velocity=clip(.92*self.velocity+.12*gate*delta/dt,-2,2)
        self.previous_obs=obs;self.last_gate=gate
        return math.exp(self.value),gate
