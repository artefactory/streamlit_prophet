FROM python:3.8-slim-buster

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN useradd -d /home/docker_user -m -s /bin/bash docker_user
USER docker_user

RUN mkdir -p /home/docker_user/workspace
WORKDIR /home/docker_user/workspace

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/install.python-poetry.org/main/install-poetry.py | POETRY_HOME=/home/docker_user/poetry python

ENV PATH="${PATH}:/home/docker_user/.poetry/bin:/home/docker_user/poetry/bin"

COPY pyproject.toml ./
COPY poetry.lock ./

RUN poetry install --no-root --no-dev

COPY . /home/docker_user/workspace/

EXPOSE 8080

ENTRYPOINT ["poetry", "run", "streamlit_prophet", "deploy", "dashboard"]
