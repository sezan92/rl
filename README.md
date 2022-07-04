# RL study Repository

### Pre-requisits
Installed softwares
- Docker
- docker-compose cli.

## Reinforce

### Build image
```sh
bash reinforce_image.sh
```
### Run the container with gpu
```sh
docker run --rm --gpus all -v $(pwd)/tests:/tests -it reinforce:latest 
```

#### Discrete on Lunarlander-v2 Environment
***train***
```sh
python3 /src/reinforce_discrete.py LunarLander-v2 --train 
```
***infer***
- setup the environment outside docker container
    ```sh
    xhost +

    ```
- If it fails and shows the message, `access control disabled, clients can connect from any host`  , then run 
    ```sh
    xhost +SI:localuser:sezan
    xhost +SI:localuser:root
    ```

- then run the docker container,
    ```sh
    docker compose up
    ```
- run the container using `docker exec -it <container_name>` in another terminal 
- inside the container run,

    ```sh
    python3 /src/reinforce_discrete.py LunarLander-v2 --infer --infer_weight /path/to/saved/weight
    ```
***test***
```sh
pytest /tests/test_*.py
```
