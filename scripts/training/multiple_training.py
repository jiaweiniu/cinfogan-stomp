import json
import train
import os

if __name__ == '__main__':
    with open(os.path.join("conf.json")) as fd:
        json_data = json.load(fd)
    configuration=json_data

    for i in range(10):
        train.training(configuration,i)
    
