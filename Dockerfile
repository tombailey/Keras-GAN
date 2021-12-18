FROM python:3.8

WORKDIR /app

COPY LICENSE .
COPY README.md .

# dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

# code
COPY ml/ ml/
COPY api.py .

# pretrained models
COPY generator.model .
COPY discriminator.model .

EXPOSE 8080

CMD ["python", "api.py"]
