from .common import STOP, STOP_MESSAGE, operating_system, get_hparams
from .carla import CarlaEnv, carla as carla_simulator, get_exec_command as get_carla_exec_command, kill_processes as kill_carla_processes, start as start_carla, restart as restart_carla, ACTIONS, ACTIONS_NAMES, CarlaEnvSettings, CARLA_SETTINGS_STATE, CARLA_SETTINGS_STATE_MESSAGE
from .tensorboard import TensorBoard
from .agent import ARTDQNAgent, AGENT_STATE, AGENT_STATE_MESSAGE, run as run_agent, play as play_agent, AGENT_IMAGE_TYPE
from .trainer import ARTDQNTrainer, TRAINER_STATE, TRAINER_STATE_MESSAGE, run as run_trainer, check_weights_size
from .console import ConsoleStats
from .commands import Commands
from . import models

import settings
settings.AGENT_IMG_TYPE = getattr(AGENT_IMAGE_TYPE, settings.AGENT_IMG_TYPE)
if models.MODEL_NAME_PREFIX:
    settings.MODEL_NAME = models.MODEL_NAME_PREFIX + ('_' if models.MODEL_NAME else '') + settings.MODEL_NAME
