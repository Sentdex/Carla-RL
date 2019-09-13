# Carla environment settings
CARLA_PATH = '../CARLA_0.9.6_Python_3.7'  # Path to Carla root folder
CARLA_HOSTS_TYPE = 'local'  # 'local' or 'remote', 'local' means that script can start and restart Carla Simulator
CARLA_HOSTS_NO = 1
CARLA_HOSTS = [['localhost', 2000, 10], ['localhost', 2002, 10]]  # List of hosts and ports and worlds to use, at least 2 ports of difference as Carla uses N and N+1 port, Town01 to Town97 for world currently, Town01 to Town07 for world are currently available, int number instead - random world change interval in minutes
SECONDS_PER_EPISODE = 10
EPISODE_FPS = 60  # Desired
IMG_WIDTH = 480
IMG_HEIGHT = 270
CAR_NPCS = 50
RESET_CAR_NPC_EVERY_N_TICKS = 1  # Resets one car NPC every given number of ticks, tick is about a second
ACTIONS = ['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']  # ['forward', 'left', 'right', 'forward_left', 'forward_right', 'backwards', 'backwards_left', 'backwards_right']
WEIGHT_REWARDS_WITH_EPISODE_PROGRESS = False  # Linearly weights rewards from 0 to 1 with episode progress (from 0 up to SECONDS_PER_EPISODE)
WEIGHT_REWARDS_WITH_SPEED = 'linear'  # 'discrete': -1 < 50kmh, 1 otherwise, 'linear': -1..1 with 0..100kmh, 'quadratic': -1..1 with 0..100kmh with formula: (speed / 100) ** 1.3 * 2 - 1
SPEED_MIN_REWARD = -1
SPEED_MAX_REWARD = 1
PREVIEW_CAMERA_RES = [[640, 400, -5, 0, 2.5], [1280, 800, -5, 0, 2.5]]  # Available resolutions from "above the car" preview camera [width, height, x, y, z], where x, y and z are related to car position
COLLISION_FILTER = [['static.sidewalk', -1], ['static.road', -1], ['vehicle.', 500]]  # list of pairs: agent id (can be part of the name) and impulse value allowed (-1 - disable collision detection entirely)

# Agent settings
AGENTS = 1
AGENT_MEMORY_FRACTION = 0.1
AGENT_GPU = None  # None, a number (to use given GPU for all agents) or a list - example [0, 1, 1] (first agent - GPU 0, 2nd and 3rd GPU 1)
AGENT_CARLA_INSTANCE = []  # Empty list for first Carla instance or list in size of AGENTS with Carla instance bounds for agents, for excample [1, 1, 2, 2]
UPDATE_WEIGHTS_EVERY = 0  # How frequently to update weights (compared to trainer fits), 0 for episode start only
AGENT_SHOW_PREVIEW = []  # List of agent id's so show a preview, or empty list
AGENT_SYNCED = True  # Synchronizes agent with frame updates from Carla
AGENT_IMG_TYPE = 'grayscaled'  # 'rgb', 'grayscaled' or 'stacked' (stacks last 3 consecutive grayscaled frames)
AGENT_ADDITIONAL_DATA = ['kmh']  # What additional data to include next to image data in observation space, possible values: kmh

# Trainer settings
MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
PREDICTION_BATCH_SIZE = 1  # How many samples to predict at once (the more, the faster)
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 2  # How many samples to fit at once (the more, the faster) - should be MINIBATCH_SIZE divided by power of 2
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
MODEL_NAME = '5_residual_#CNN_KERNELS#'  # model name, prefixed from sources/models.py, #MODEL_ARCHITECTURE# adds model architectore acronym, #CNN_KERNELS# adds number of kernels from all CNN layers
MIN_REWARD = 100  # For model save
TRAINER_MEMORY_FRACTION = 0.6
TRAINER_GPU = None  # None - not set, 0, 1, ... - GPU with given index
SAVE_CHECKPOINT_EVERY = 100  # episodes

# DQN settings
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 20_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 5_000  # Minimum number of steps in a memory to start training

# Exploration settings
START_EPSILON = 1
EPSILON_DECAY = 0.99995  # 0.99975
MIN_EPSILON = 0.1

# Model settings
MODEL_BASE = '5_residual_CNN'  # from models.py
MODEL_HEAD = 'hidden_dense'  # from models.py
MODEL_SETTINGS = {'hidden_1_units': 256}  # 'hidden_1_units': 1024 for Xception

# Optimizer settings
OPTIMIZER_LEARNING_RATE = 0.001
OPTIMIZER_DECAY = 0.0

# Conv Cam
CONV_CAM_LAYER = 'auto_act'  # 'auto' - finds and uses last activation layer, 'auto_act' - uses Activation layer after last convolution layer if exists
CONV_CAM_AGENTS = [1]

# Console settings
SHOW_CARLA_ENV_SETTINGS = False
