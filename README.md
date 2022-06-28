# RL study Repository

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
python3 reinforce/reinforce_discrete.py LunarLander-v2 --train 
```
***infer***
```sh
python3 reinforce/reinforce_discrete.py LunarLander-v2 --infer --infer_weight /path/to/saved/weight
```
***test***
```sh
pytest /tests/test_*.py
```
