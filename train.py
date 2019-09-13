import os
import time
from collections import deque
from multiprocessing import Process, Value, Array, Queue
from threading import Thread
import subprocess

import settings
from sources import start_carla, restart_carla
from sources import STOP, get_hparams
from sources import run_agent, AGENT_STATE
from sources import run_trainer, check_weights_size, TRAINER_STATE, CarlaEnv
from sources import ConsoleStats, Commands
from sources import get_carla_exec_command, kill_carla_processes, CarlaEnvSettings, CARLA_SETTINGS_STATE


if __name__ == '__main__':
    
    print('Starting...')

    # overal start time
    start_time = time.time()

    # Create required folders
    os.makedirs('models', exist_ok=True)
    os.makedirs('tmp', exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)

    # Kill Carla processes if there are any and start simulator
    start_carla()

    # Load hparams if they are being saved by trainer
    hparams = get_hparams()
    if hparams:
        # If everything is ok, update start time by previous running time
        start_time -= hparams['duration']

    # Spawn limited trainer process and get weights' size
    print('Calculating weights size...')
    weights_size = Value('L', 0)
    p = Process(target=check_weights_size, args=(hparams['model_path'] if hparams else False, weights_size), daemon=True)
    p.start()
    while weights_size.value == 0:
        time.sleep(0.01)
    p.join()

    # A bunch of variabled and shared variables used to set all parts of ARTDQN and communicate them
    duration = Value('d')
    episode = Value('L', hparams['episode'] if hparams else 0)
    epsilon = Array('d', hparams['epsilon'] if hparams else [settings.START_EPSILON, settings.EPSILON_DECAY, settings.MIN_EPSILON])
    discount = Value('d', hparams['discount'] if hparams else settings.DISCOUNT)
    update_target_every = Value('L', hparams['update_target_every'] if hparams else settings.UPDATE_TARGET_EVERY)
    last_target_update = hparams['last_target_update'] if hparams else 0
    min_reward = Value('f', hparams['min_reward'] if hparams else settings.MIN_REWARD)
    agent_show_preview = []
    for agent in range(settings.AGENTS):
        if hparams:
            agent_show_preview.append(Array('f', hparams['agent_show_preview'][agent]))
        else:
            agent_show_preview.append(Array('f', [(agent + 1) in settings.AGENT_SHOW_PREVIEW, 0, 0, 0, 0, 0]))
    save_checkpoint_every = Value('L', hparams['save_checkpoint_every'] if hparams else settings.SAVE_CHECKPOINT_EVERY)
    seconds_per_episode = Value('L', hparams['seconds_per_episode'] if hparams else settings.SECONDS_PER_EPISODE)
    weights = Array('c', weights_size.value)
    weights_iteration = Value('L', hparams['weights_iteration'] if hparams else 0)
    transitions = Queue()
    tensorboard_stats = Queue()
    trainer_stats = Array('f', [0, 0])
    carla_check = None
    episode_stats = Array('d', [-10**6, -10**6, -10**6, 0, 0, 0, 0, -10**6, -10**6, -10**6] + [-10**6 for _ in range((CarlaEnv.action_space_size + 1) * 3)])
    stop = Value('B', 0)
    agent_stats = []
    for _ in range(settings.AGENTS):
        agent_stats.append(Array('f', [0, 0, 0]))
    optimizer = Array('d', [-1, -1, 0, 0, 0, 0])
    car_npcs = Array('L', hparams['car_npcs'] if hparams else [settings.CAR_NPCS, settings.RESET_CAR_NPC_EVERY_N_TICKS])
    pause_agents = []
    for _ in range(settings.AGENTS):
        pause_agents.append(Value('B', 0))

    # Run Carla settings (weather, NPC control) in a separate thread
    carla_settings_threads = []
    carla_settings_stats = []
    carla_frametimes_list = []
    carla_fps_counters = []
    carla_fps = []
    agents_in_carla_instance = {}
    for process_no in range(settings.CARLA_HOSTS_NO):
        agents_in_carla_instance[process_no] = []
    for agent in range(settings.AGENTS):
        carla_instance = 1 if not len(settings.AGENT_CARLA_INSTANCE) or settings.AGENT_CARLA_INSTANCE[agent] > settings.CARLA_HOSTS_NO else settings.AGENT_CARLA_INSTANCE[agent]
        agents_in_carla_instance[carla_instance-1].append(pause_agents[agent])
    for process_no in range(settings.CARLA_HOSTS_NO):
        carla_settings_process_stats = Array('f', [-1, -1, -1, -1, -1, -1])
        carla_frametimes = Queue()
        carla_frametimes_list.append(carla_frametimes)
        carla_fps_counter = deque(maxlen=60)
        carla_fps.append(Value('f', 0))
        carla_fps_counters.append(carla_fps_counter)
        carla_settings_stats.append(carla_settings_process_stats)
        carla_settings = CarlaEnvSettings(process_no, agents_in_carla_instance[process_no], stop, car_npcs, carla_settings_process_stats)
        carla_settings_thread = Thread(target=carla_settings.update_settings_in_loop, daemon=True)
        carla_settings_thread.start()
        carla_settings_threads.append([carla_settings_thread, carla_settings])

    # Start trainer process
    print('Starting trainer...')
    trainer_process = Process(target=run_trainer, args=(hparams['model_path'] if hparams else False, hparams['logdir'] if hparams else False, stop, weights, weights_iteration, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, transitions, tensorboard_stats, trainer_stats, episode_stats, optimizer, hparams['models'] if hparams else [], car_npcs, carla_settings_stats, carla_fps), daemon=True)
    trainer_process.start()

    # Wait for trainer to be ready, it needs to, for example, dump weights that agents are going to update
    while trainer_stats[0] != TRAINER_STATE.waiting:
        time.sleep(0.01)

    # Start one new process for each agent
    print('Starting agents...')
    agents = []
    for agent in range(settings.AGENTS):
        carla_instance = 1 if not len(settings.AGENT_CARLA_INSTANCE) or settings.AGENT_CARLA_INSTANCE[agent] > settings.CARLA_HOSTS_NO else settings.AGENT_CARLA_INSTANCE[agent]
        p = Process(target=run_agent, args=(agent, carla_instance-1, stop, pause_agents[agent], episode, epsilon, agent_show_preview[agent], weights, weights_iteration, transitions, tensorboard_stats, agent_stats[agent], carla_frametimes_list[carla_instance-1], seconds_per_episode), daemon=True)
        p.start()
        agents.append(p)

    print('Ready')

    # Start printing stats to a console
    print('\n'*(settings.AGENTS+22))
    console_stats = ConsoleStats(stop, duration, start_time, episode, epsilon, trainer_stats, agent_stats, episode_stats, carla_fps, weights_iteration, optimizer, carla_settings_threads, seconds_per_episode)
    console_stats_thread = Thread(target=console_stats.print, daemon=True)
    console_stats_thread.start()

    # Create commands' object
    commands = Commands(stop, epsilon, discount, update_target_every, min_reward, save_checkpoint_every, seconds_per_episode, agent_show_preview, optimizer, car_npcs)

    # Main loop
    while True:

        # If everything is running or carla broke...
        if stop.value in[STOP.running, STOP.carla_simulator_error, STOP.restarting_carla_simulator, STOP.carla_simulator_restarted]:

            # ...and all agents return an error
            if any([state[0] == AGENT_STATE.error for state in agent_stats]):

                # If it's a running state, set it to carla error
                if stop.value == STOP.running:
                    stop.value = STOP.carla_simulator_error
                    for process_no in range(settings.CARLA_HOSTS_NO):
                        carla_fps_counters[process_no].clear()

            # If agents are not returning errors, set running state
            else:
                stop.value = STOP.running
                carla_check = None

        # Append new frametimes from carla for stats
        if not stop.value == STOP.carla_simulator_error:
            for process_no in range(settings.CARLA_HOSTS_NO):
                for _ in range(carla_frametimes_list[process_no].qsize()):
                    try:
                        carla_fps_counters[process_no].append(carla_frametimes_list[process_no].get(True, 0.1))
                    except:
                        break
                carla_fps[process_no].value = len(carla_fps_counters[process_no]) / sum(carla_fps_counters[process_no]) if sum(carla_fps_counters[process_no]) > 0 else 0

        # If carla broke
        if stop.value == STOP.carla_simulator_error and settings.CARLA_HOSTS_TYPE == 'local':

            # First check, set a timer because...
            if carla_check is None:
                carla_check = time.time()

            # ...we give it 15 seconds to possibly recover, if not...
            if time.time() > carla_check + 15:

                # ... set Carla restart state and try to restart it
                stop.value = STOP.restarting_carla_simulator
                if settings.CARLA_HOSTS_TYPE == 'local':
                    kill_carla_processes()
                for process_no in range(settings.CARLA_HOSTS_NO):
                    carla_settings_threads[process_no][1].clean_carnpcs()
                    carla_settings_threads[process_no][1].restart = True
                    carla_fps_counters[process_no].clear()
                    carla_fps[process_no].value = 0
                for process_no in range(settings.CARLA_HOSTS_NO):
                    while not carla_settings_threads[process_no][1].state == CARLA_SETTINGS_STATE.restarting:
                        time.sleep(0.1)
                restart_carla()
                for process_no in range(settings.CARLA_HOSTS_NO):
                    carla_settings_threads[process_no][1].restart = False
                stop.value = STOP.carla_simulator_restarted

        # When Carla restarts, give it up to 60 seconds, then try again if failed
        if stop.value == STOP.restarting_carla_simulator and time.time() > carla_check + 60:
            stop.value = STOP.carla_simulator_error
            carla_check = time.time() - 15

        # Process commands
        commands.process()

        # If stopping - cleanup and exit
        if stop.value == STOP.stopping:

            # Trainer process already "knows" that, just wait for it to exit
            trainer_process.join()

            # The same for all agents
            for agent in agents:
                agent.join()

            # ... and Carla settings
            for process_no in range(settings.CARLA_HOSTS_NO):
                carla_settings_threads[process_no][0].join()

            # Close Carla
            kill_carla_processes()

            stop.value = STOP.stopped
            time.sleep(1)
            break

        time.sleep(0.01)
