FROM python:3.8

WORKDIR /app

COPY LICENSE .
COPY README.md .

# dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

# code
COPY ml/ ml/
COPY api.py .

# pretrained models
COPY output/unet/ output/unet/
EXPOSE 8080

CMD ["python", "api.py"]
