# syntax=docker/dockerfile:1.4

FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder
EXPOSE 8000
WORKDIR /app
COPY requirements.txt /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && pip3 install -r requirements.txt --no-cache-dir \
    && apt-get remove -y gcc python3-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
COPY . /app
ENTRYPOINT ["python3"]
CMD ["manage.py", "runserver", "0.0.0.0:8000"]

FROM builder as dev-envs
RUN apt-get update && apt-get install -y git

RUN groupadd docker \
    && useradd -g docker vscode

# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /
CMD ["manage.py", "runserver", "0.0.0.0:8000"]