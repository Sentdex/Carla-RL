from sources import play_agent, start_carla, get_hparams, ConsoleStats, CarlaEnvSettings
import os
from threading import Thread

# Replacement for shared multiprocessing object when playing
class PauseObject:
    value = 0

if __name__ == '__main__':
    
    # Load hparams if they are being saved by trainer
    hparams = get_hparams(playing=True)

    print('Saved models:')

    # Check paths from hparams, filter out models whose files exist on disk
    filtered_models = {}
    for model in hparams['models']:
        if os.path.isfile(model):
            filtered_models[int(model.split('.')[0].split('_')[-1])] = model
    filtered_models = list(dict(sorted(filtered_models.items(), key=lambda kv: kv[0])).values())

    for index, model in enumerate(filtered_models):
        print(f'{index+1}. {model}')

    # Ask user for a model to choose
    model = filtered_models[int(input('Choose the model (empty - most recent one): ') or '0') - 1]

    print('Starting...')

    # Kill Carla processes if there are any and start simulator
    start_carla(playing=True)

    # Run Carla settings (weather, NPC control) in a separate thread
    pause_object = PauseObject()
    carla_settings = CarlaEnvSettings(0, [pause_object], car_npcs=hparams['car_npcs'])
    carla_settings_thread = Thread(target=carla_settings.update_settings_in_loop, daemon=True)
    carla_settings_thread.start()

    # Play
    print(f'Starting agent ({model})...')
    play_agent(model, pause_object, ConsoleStats.print_short)
