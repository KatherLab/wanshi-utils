FROM python:3 AS base
WORKDIR /opt/wanshi
COPY pyproject.toml setup.cfg /opt/wanshi
RUN pip install /opt/wanshi

FROM base AS develop
RUN set -eux; \
	apt-get update; \
	apt-get install -y vim; \
	pip install black isort mypy types-tqdm
ENTRYPOINT [ "bash" ]

FROM base AS deploy
COPY . /opt/wanshi
ENV PYTHONPATH /opt/wanshi
ENTRYPOINT ["python3"]
