"""Decode TH14 1.00b collectible candidates from the original game's item pool."""
import numpy as np


class ItemReader:
    def __init__(self, reader):
        self.reader=reader
        self.layout=reader.offsets.items
        self.types=reader.offsets.associations.item_types
        o=self.layout
        self.dtype=np.dtype(dict(names=["position","velocity","age","state","kind"],
            formats=[("<f4",2),("<f4",2),"<i4","<i4","<i4"],
            offsets=[o.zItem_pos,o.zItem_vel,o.zItem_timer,o.zItem_state,o.zItem_type],itemsize=o.zItem_len))
        if o.zItemManager_array_len*o.zItem_len > 16*1024**2:
            raise ValueError("Item pool is outside the verified TH14 size.")

    def visible(self):
        o=self.layout
        manager=self.reader.read_uint(o.item_manager_pointer,relative=True)
        if not manager:
            raise ValueError("Original item manager is unavailable.")
        n=o.zItemManager_array_len
        # One local call replaces thousands of per-slot system calls. Keep only a
        # compact copy of active, recognized slots after decoding the strided pool.
        raw=self.reader.game._read_game_memory(manager+o.zItemManager_array,n*o.zItem_len,rel=False)
        pool=np.frombuffer(raw,dtype=self.dtype,count=n)
        slots=np.flatnonzero(pool["state"]!=0)
        selected=pool[slots]
        valid=np.isin(selected["kind"],list(self.types))
        s=selected[valid]
        compact=np.column_stack((slots[valid],s["kind"],s["state"],s["position"],s["velocity"],s["age"]+1))
        if not np.isfinite(compact).all():
            raise ValueError("Native item observation contained non-finite positions.")
        return compact.astype(np.float32,copy=False),int((~valid).sum())

    @staticmethod
    def tokens(items,player,limit=24):
        # Reserve an equal opportunity to see life/Bomb/power/point types by
        # selecting the nearest representative of each visible native type first.
        # Remaining slots are filled by proximity. No movement command is derived.
        output=np.zeros((limit,8),np.float32)
        xy=items[:,3:5]-player
        order=np.argsort((xy**2).sum(1),kind="stable")
        _,first=np.unique(items[order,1],return_index=True)
        chosen=np.r_[order[np.sort(first)],np.delete(order,first)][:limit]
        if len(chosen):
            s=items[chosen]
            output[:len(s)]=np.column_stack((xy[chosen]/[384,448],s[:,5:7]/16,
                s[:,1],s[:,2],np.minimum(s[:,7]/600,2),np.ones(len(s))))
        return output
