# Unsloth on DGX Spark

1. Enter the DGX
    ```bash
    ssh <user>@<IP>
    ```
2. Clone the repo
    ```bash
    git clone https://github.com/riomus/dgx-spark-unsloth.git
    ```
3. Run NVidia image

Check for latest NVidia pytorch version in [their registry](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)


   ```bash
   docker run --gpus all --ulimit memlock=-1 -it --ulimit stack=67108864 -it --entrypoint /usr/bin/bash  -v "$PWD"/dgx-spark-unsloth:/workspace -p 8888:8888  --rm nvcr.io/nvidia/pytorch:25.10-py3 
   ```
4. Install UV
5. Create venv and install project

```bash
    sh -c install.sh
```

6. Start Jupyter lab
   

```bash
    uv run jupyter lab
```

7. Port-forward/use Nvidia sync - and enter the Jupter Lab
