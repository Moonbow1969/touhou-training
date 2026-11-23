FROM python:3.12-slim-bookworm

ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install --no-cache-dir gymnasium==1.0.0 stable-baselines3==2.5.0

ENTRYPOINT ["python", "-u"]
