FROM nvcr.io/nvidia/pytorch:25.10-py3

COPY install.sh pyproject.toml uv.lock python-version .

ENV VIRTUAL_ENV=/venv
RUN mkdir /venv && bash install.sh
ENV PATH=/venv/bin:$PATH
EXPOSE 2718
