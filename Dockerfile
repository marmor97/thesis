FROM python:3.9-slim-buster

COPY requirements.txt .
COPY requirements.cpu.txt .

# Added
RUN apt-get update
RUN apt-get install -y wget

# Upgrade sqlite3
RUN apt-get install --only-upgrade sqlite3

# Install yq (a lightweight and portable command-line YAML processor)
ARG VERSION=v4.27.5
ARG BINARY=yq_linux_386
RUN wget https://github.com/mikefarah/yq/releases/download/${VERSION}/${BINARY} -O /usr/bin/yq
RUN chmod +x /usr/bin/yq

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install llm-foundry==0.5.0
RUN pip install transformers==4.37.0
RUN pip install wandb==0.16.4
RUN pip install langchain-core==0.1.30
RUN pip install langchain==0.1.10
# RUN pip install langchain-community==0.0.27
RUN pip install chromadb==0.4.24
RUN pip install faiss-cpu
RUN pip install tqdm
RUN pip install nltk==3.8.1
RUN pip install rouge_score==0.1.2
RUN pip install scikit-learn
RUN pip install -r requirements.txt && pip install -r requirements.cpu.txt

# Copy application code
COPY . /code
WORKDIR /code

# Set the entrypoint
ENTRYPOINT ["bash", "/code/scripts/train_entrypoint.sh"]