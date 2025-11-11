FROM nvcr.io/nvidia/pytorch:25.10-py3

COPY install.sh pyproject.toml uv.lock python-version .

ENV VIRTUAL_ENV=/venv
RUN mkdir /venv && bash install.sh
ENV PATH=/venv/bin:/root/.local/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV UV_NO_PROJECT=1
RUN apt update && apt install just && apt clean
EXPOSE 2718
