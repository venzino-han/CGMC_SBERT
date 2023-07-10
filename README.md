## Just accepted, IEEE ICWS 2023

# Amazon Review Dataset 
- Original dataset from : https://jmcauley.ucsd.edu/data/amazon/
- Pre-processed dataset in `dataset` directory. 
<br />

# Docker Container
- Docker container use cgmc project directory as volume 
- File change will be apply directly to file in docker container

# Train 
1. `make up` : build docker image and start docker container
2. check `train_config/train_list.ymal` file 
3. `python3 src/train.py` : start train in docker container

# Evaluation 
1. `make up` : build docker image and start docker container
2. check `test_config/test_list.ymal` file 
3. `python3 src/evaluate.py` : start evaluation in docker container
4. you can check the result in `log/test_rmse.log`

<br />
