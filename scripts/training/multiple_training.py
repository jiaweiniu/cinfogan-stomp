import json
import train
import os
from tqdm import tqdm

if __name__ == '__main__':
    with open(os.path.join("conf.json")) as fd:
        json_data = json.load(fd)
    configuration=json_data

    # CGAN training
    configuration["cinfogan"]=False 
    for i in tqdm(range(10)):
        train.training(configuration,i)

    # CInfoGan training 
    configuration["cinfogan"]=True 
    for i in tqdm(range(10)):
        train.training(configuration,i)
