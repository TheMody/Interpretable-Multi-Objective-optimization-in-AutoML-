
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime as dt
from train import train
import configparser
from logger import Logger
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--config_file', help='path_to_config_file', type=str, default="config.json")
    parser.add_argument('--test', help='whether to run prelim test or not', action="store_true", default=False)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.sections()
    config.read(args.config_file)
    if config["DEFAULT"]["directory"] == "default":
        config["DEFAULT"]["directory"] = "results/" + dt.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    else:
        config["DEFAULT"]["directory"] = "results/" + config["DEFAULT"]["directory"] 

    os.makedirs(config["DEFAULT"]["directory"], exist_ok=True)
    print(config["DEFAULT"]["directory"])
    
    
    
    for file in os.listdir(os.getcwd()):
        if ".py" in file or ".json" in file:
            shutil.copy2(file, config["DEFAULT"]["directory"] )
            
    sys.stdout = Logger(open(config["DEFAULT"]["directory"] + "/SysOut.txt", "w"))
    print("config file used: ",args.config_file)
    train(args, config)


if __name__ == '__main__':
    main()

