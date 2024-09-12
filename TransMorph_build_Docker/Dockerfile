FROM python:3.9-slim

RUN python -m pip install --user -U pip

RUN mkdir -p /input /output

COPY ./pretrained_weights pretrained_weights/
COPY ./requirements.txt .
COPY ./infer_TransMorph.py .
COPY ./TransMorph.py .
COPY ./configs_TransMorph.py .
COPY ./losses.py .
COPY ./template.nii.gz .

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --user -r requirements.txt
RUN pip install antspyx
RUN pip install intensity-normalization

ENTRYPOINT ["python3","-u","./infer_TransMorph.py"]