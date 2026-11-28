"""Bounded full-color video recording on its own Linux CPU affinity."""
import os
from queue import Full, Queue
from threading import Event, Thread
import time
import cv2


class MovieQueue:
    def __init__(self,path,core):
        self.path=path;self.core=core;self.queue=Queue(maxsize=16);self.ready=Event()
        self.error=None;self.written=0;self.depth=0;self.backpressure_ns=0
        self.thread=Thread(target=self.encode,name="th08-movie",daemon=True)
        self.thread.start();self.ready.wait()
        if self.error:raise self.error

    def encode(self):
        encoder=None
        try:
            os.sched_setaffinity(0,{self.core})
            encoder=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*"mp4v"),8.,(640,480))
            if not encoder.isOpened():raise RuntimeError("Native movie encoder unavailable")
            self.ready.set()
            while (frame:=self.queue.get()) is not None:
                encoder.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                self.written+=1
        except Exception as exc:self.error=exc
        finally:
            if encoder is not None:encoder.release()
            self.ready.set()

    def submit(self,frame):
        if self.error:raise self.error
        try:self.queue.put_nowait(frame)
        except Full:
            started=time.monotonic_ns()
            while self.thread.is_alive():
                try:self.queue.put(frame,timeout=.25);break
                except Full:continue
            self.backpressure_ns+=time.monotonic_ns()-started
        if self.error:raise self.error
        self.depth=max(self.depth,self.queue.qsize())

    def close(self):
        self.submit(None)
        self.thread.join()
        if self.error:raise self.error
        return dict(encoded_frames=self.written,peak_queue_depth=self.depth,
                    bound_frames=self.queue.maxsize,backpressure_ns=self.backpressure_ns,cpu=self.core)
