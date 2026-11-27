"""Chronology for native TH17 per-stage clocks. Chapter and music clocks may restart."""
import numpy as np

def coordinates(stages, ticks):
    stages = np.asarray(stages,dtype=np.int64)
    ticks = np.asarray(ticks,dtype=np.int64)
    if len(stages)!=len(ticks) or not len(stages) or np.any((stages<1)|(stages>3)) or np.any((ticks<0)|(ticks>=2**31)):
        raise ValueError("Invalid native stage or stage tick")
    change=np.diff(stages)
    if np.any((change<0)|(change>1)) or np.any((change==0)&(np.diff(ticks)<0)):
        raise ValueError("Stage skipped or native stage clock decreased inside a stage")
    return (stages<<32)+ticks

def progressed_ticks(stages, ticks):
    coordinates(stages,ticks)
    ticks=np.asarray(ticks,dtype=np.int64)
    changes=np.where(np.diff(stages)==0,np.diff(ticks),ticks[1:])
    return np.r_[0,np.cumsum(changes)]

def event_coordinate(stage, tick):
    return int(stage)*(1<<32)+int(tick)
