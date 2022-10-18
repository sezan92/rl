# RL study Repository

### Pre-requisits
Installed softwares
- Docker
- docker-compose cli.

## Reinforce

### Build image
```sh
docker compose build
```
### Run the container with gpu
```sh
docker compose up 
```

#### Discrete on Lunarlander-v2 Environment

***train***
- Run the container using `docker exec -it <container_name>` in another terminal
- From inside the container run,
    ```sh
    python3 /src/reinforce_discrete.py LunarLander-v2 --train --save_model_path </path/to/save/the/model> --gamma <gamma hyper-parameter> --epoch <num_of_epoch> --plot <to plot or not> --plot_fig_path </path/to/save/the/plot>
    ```

***infer***
- setup the environment outside docker container
    ```sh
    sudo xhost +SI:localuser:<username>

    ```
- If it fails and shows the message, `access control disabled, clients can connect from any host`  , then run 
    ```sh
    xhost +SI:localuser:<username>
    ```

- run the container using `docker exec -it <container_name>` in another terminal 
- inside the container run,

    ```sh
    python3 /src/reinforce_discrete.py LunarLander-v2 --infer --infer_weight /path/to/saved/weight --infer_render <to render the inference or not> --infer_render_fps <fps for render video> --infer_video </path/to/save/inference/rendered/video.>
    ```
***test***
#### TODO (cover all behaviour)
```sh
pytest /tests/test_*.py
```
