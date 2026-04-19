import yaml
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

def load_config():
    with open("/mnt/c/Users/Artur/Desktop/agh/cuda/dlcuda/lab1/refactored_model_images/config.yaml") as stream:
        return yaml.safe_load(stream)
    
def manage_config(config):
    new_config = deepcopy(config)
    while True:
        print("If you want to progress futher, simply type 'continue'. If you want to change any existing parameter, write it in format 'parameter_name:new_value'. If you want to start once again, write 'restart'. Every other input will be ignored.")
        user_input = input("Your input: ")
        if user_input.lower().strip() == "continue":
            break
        if user_input.lower().strip() == "restart":
            new_config = config.deepcopy()
            continue
        try:
            parameter, value = map(str.strip, user_input.split(":"))
            if parameter not in config:
                raise ValueError()
        except ValueError:
            print("Invalid input format. Please use 'parameter_name:new_value'.")
            continue
        print(f"Changing {parameter} from {config[parameter]} to {value}.")
        new_config[parameter] = type(config[parameter])(value)
    return new_config

def get_writer():
    return SummaryWriter(log_dir="runs/baseline_experiment")  #tb

