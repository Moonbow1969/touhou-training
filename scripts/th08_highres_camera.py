"""Preserve 192×224 grayscale TH08 observations on a separate evidence CPU."""
import os

import cv2
import numpy as np

from scripts.th08_movie_queue import MovieQueue


class HighResolutionQueue(MovieQueue):
    def encode(self):
        try:
            os.sched_setaffinity(0,{self.core})
            cv2.setNumThreads(1)
            self.ready.set()
            frames=[]
            while (frame:=self.queue.get()) is not None:
                area=cv2.resize(frame[16:464,32:416],(192,224),interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(area,cv2.COLOR_RGB2GRAY))
                self.written+=1
            np.savez_compressed(self.path,gray=np.stack(frames))
        except Exception as exc:
            self.error=exc
        finally:
            self.ready.set()
