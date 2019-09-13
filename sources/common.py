from dataclasses import dataclass
import os
import json
import settings


# Stop reasons
@dataclass
class STOP:
    running = 0
    now = 1
    at_checkpoint = 2
    stopping = 3
    stopped = 4
    carla_simulator_error = 5
    restarting_carla_simulator = 6
    carla_simulator_restarted = 7


# Stop state messages
STOP_MESSAGE = {
    0: 'RUNNING',
    1: 'STOP NOW',
    2: 'STOP AT CHECKPOINT',
    3: 'STOPPING',
    4: 'STOPPED',
    5: 'CARLA SIMULATOR ERROR',
    6: 'RESTART' + ('ING' if settings.CARLA_HOSTS_TYPE == 'local' else '') + ' CARLA SIMULATOR',
    7: 'CARLA SIMULATOR RESTARTED',
}

def operating_system():
    return 'windows' if os.name == 'nt' else 'linux'


# Tries to load and return hparams
def get_hparams(playing=False):

    hparams = None
    if os.path.isfile('checkpoint/hparams.json'):

        # If JSON file exists, load it
        with open('checkpoint/hparams.json', encoding='utf-8') as f:
            hparams = json.load(f)

        # But if model file does not exist, remove hparams (only when training)
        if not os.path.isfile(hparams['model_path']) and not playing:
            hparams = None

    return hparams
