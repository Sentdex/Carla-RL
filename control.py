import os
import time


# Prints help message
def print_help():
    #print('\n')
    print()
    print('Commands:')
    print('epsilon current|decay|min value       - set current epsilon, epsilon decay or minimum')
    print('                                        value to keep, value - float in range 0.0 to 1.0')
    print('discount value                        - set dicount value - float in range 0.0 to 1.0')
    print('target update_every value             - set target network update frequency')
    print('                                        to value (int) number of episodes')
    print('reward min value                      - set minimum mean value (int) of reward to save a model')
    print('checkpoint save_every value           - set checkpoint frequency (int)')
    print('episode duration value                - set episode duration to value (int) seconds')
    print('preview id on|env|agent|cam_x|off     - turn on/off agent preview,')
    print('                                        id - int, agent number or "all",')
    print('                                        on/env - see what environment returns,')
    print('                                        agent - see what agent sees,')
    print('                                        cam_x - predefined camera (x - int starting from 1)')
    print('                                        px,py,cx,cy,cz - own camera, px and py preview size')
    print('                                          cx, cy and cz - camera position relative to the car')
    print('stop now|checkpoint                   - stop now or on nearest checkpoint')
    print('optimizer lr|decay value              - set optimizer\'s learning rate and decay,')
    print('                                        float in range 0.0 to 1.0')
    print('carnpcs keep|reset_interval value     - number of car NPCs to keep in environment')
    print('                                        and car reset interval - int')
    print('values                                - returns current changable values')
    print('help                                  - this help')
    print()


# Reads an answer and returns it
def receive_answer():

    received_answer = False

    # Loop through all the files...
    for output_file in os.listdir('tmp'):

        # ...and open output ones only
        if not output_file.startswith('output_'):
            time.sleep(0.01)
            continue

        # Try to open and read a file, print it, then remove file
        try:
            time.sleep(0.01)
            with open('tmp/' + output_file, encoding='utf-8') as f:
                print(f.read())
                received_answer = True

            os.remove('tmp/' + output_file)
        except Exception as e:
            print(str(e))

    return received_answer


# At start print help and clean folder from previous unreaded answers (if any exist)
print_help()
receive_answer()

# Nain loop
while True:

    # Get a command
    command = input('> ')

    # If command is not empty
    if command:

        # If help, print help
        if command == 'help':
            print_help()
            continue

        # Else save it as a command for ARTDQN
        with open(f'tmp/command_{int(time.time())}', 'w', encoding='utf-8') as f:
            f.write(command)

        # Wait for an answer
        while True:
            if receive_answer():
                break
