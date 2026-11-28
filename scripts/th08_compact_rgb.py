"""Preserve an area-averaged RGB playfield for every captured TH08 frame on the evidence CPU."""
import os
import cv2
import numpy as np
from scripts.th08_movie_queue import MovieQueue


class CompactRGBQueue(MovieQueue):
    def encode(self):
        try:
            os.sched_setaffinity(0,{self.core})
            frames=[]
            self.ready.set()
            while (frame:=self.queue.get()) is not None:
                # Eight-by-eight area average, sharing exact crop coordinates with the policy camera.
                frames.append(cv2.resize(frame[16:464,32:416],(48,56),interpolation=cv2.INTER_AREA))
                self.written+=1
            np.savez_compressed(self.path,rgb=np.stack(frames))
        except Exception as exc:
            self.error=exc
        finally:
            self.ready.set()
