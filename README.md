# RL study Repository

## Reinforce

### Build image
```sh
bash reinforce_image.sh
```
### Run the container with gpu
```sh
docker run --rm --gpus all -it reinforce:latest 
```

### Discrete on Lunarlander-v2 Environment
***train***
```
python3 reinforce/reinforce_discrete.py LunarLander-v2 --train 
```
***test***
```
python3 reinforce/reinforce_discrete.py LunarLander-v2 --infer --infer_weight /path/to/saved/weight
```

### TODO
- [] Write optional arguments