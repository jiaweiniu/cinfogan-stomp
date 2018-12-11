import json
import train
import os

if __name__ == '__main__':
    with open(os.path.join("conf.json")) as fd:
        json_data = json.load(fd)
    configuration=json_data

    # CGAN training
    configuration["cinfogan"]=False 
    for i in range(10):
        train.training(configuration,i)

    # CInfoGan training 
    configuration["cinfogan"]=True 
    for i in range(10):
        train.training(configuration,i)
