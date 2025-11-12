# Unsloth on DGX Spark

This was inspired by https://github.com/riomus/dgx-spark-unsloth.git

## Quick Start

```shell
apt update && apt install just
just build
```

This will create an image named `spark-unsloth`.  You can get a shell into this container with:

```shell
docker run --gpus all --ulimit memlock=-1 -it --ulimit stack=67108864 -it --entrypoint /usr/bin/bash spark-unsloth
```

Or better yet just run `just run` from this directory, or `just unsloth-shell` from our parent directory.
