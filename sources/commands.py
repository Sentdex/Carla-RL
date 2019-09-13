import os
import time
from sources import STOP
import settings


class Commands:

    def __init__(self, stop, epsilon, discount, update_target_every, min_reward, save_checkpoint_every, seconds_per_episode, agent_show_preview, optimizer, car_npcs):
        self.stop = stop
        self.epsilon = epsilon
        self.discount = discount
        self.update_target_every = update_target_every
        self.min_reward = min_reward
        self.save_checkpoint_every = save_checkpoint_every
        self.seconds_per_episode = seconds_per_episode
        self.agent_show_preview = agent_show_preview
        self.optimizer = optimizer
        self.car_npcs = car_npcs

    def process(self):

        output = ''
        error = ''

        # Check every file...
        for command_file in os.listdir('tmp'):

            # ...whose name starts witn a command prefix...
            if not command_file.startswith('command_'):
                continue

            # ...and try to open and read
            try:

                time.sleep(0.01)

                with open('tmp/' + command_file, encoding='utf-8') as f:
                    command = f.read().split()

                    command[0] = command[0].lower()

                    # Epsilon value
                    if command[0] == 'epsilon' and command[1] == 'current':
                        value = float(command[2])
                        if value > 1 or value < 0:
                            raise ValueError(f'Epsilon value {value} out of range')
                        old_value = self.epsilon[0]
                        self.epsilon[0] = value
                        output += f'Epsilon value updated from {old_value} to {value}\n'

                    # Epsilon decay
                    elif command[0] == 'epsilon' and command[1] == 'decay':
                        value = float(command[2])
                        if value > 1 or value < 0:
                            raise ValueError(f'Epsilon decay value {value} out of range')
                        old_value = self.epsilon[1]
                        self.epsilon[1] = value
                        output += f'Epsilon decay value updated from {old_value} to {value}\n'

                    # Minimum epsilon
                    elif command[0] == 'epsilon' and command[1] == 'min':
                        value = float(command[2])
                        if value > 1 or value < 0:
                            raise ValueError(f'Minimum epsilon value {value} out of range')
                        old_value = self.epsilon[2]
                        self.epsilon[2] = value
                        output += f'Minimum epsilon value updated from {old_value} to {value}\n'

                    # Discount
                    elif command[0] == 'discount':
                        value = float(command[1])
                        if value > 1 or value < 0:
                            raise ValueError(f'Minimum discount value {value} out of range')
                        old_value = self.discount.value
                        self.discount.value = value
                        output += f'Discount value updated from {old_value} to {value}\n'

                    # Target network update interval
                    elif command[0] == 'target' and command[1] == 'update_every':
                        value = int(command[2])
                        if value < 0:
                            raise ValueError(f'Target network update every value {value} out of range')
                        old_value = self.update_target_every.value
                        self.update_target_every.value = value
                        output += f'Target network update every value updated from {old_value} to {value}\n'

                    # Minimum reward for model saving
                    elif command[0] == 'reward' and command[1] == 'min':
                        value = int(command[2])
                        old_value = self.min_reward.value
                        self.min_reward.value = value
                        output += f'Minimum reward value updated from {old_value} to {value}\n'

                    # Checkpoint save interval
                    elif command[0] == 'checkpoint' and command[1] == 'save_every':
                        value = int(command[2])
                        if value < 0:
                            raise ValueError(f'Save checkpoint every value {value} out of range')
                        old_value = self.save_checkpoint_every.value
                        self.save_checkpoint_every.value = value
                        output += f'Save checkpoint every value updated from {old_value} to {value}\n'

                    # Episode duration time
                    elif command[0] == 'episode' and command[1] == 'duration':
                        value = int(command[2])
                        if value < 0:
                            raise ValueError(f'Episode duration value {value} out of range')
                        old_value = self.seconds_per_episode.value
                        self.seconds_per_episode.value = value
                        output += f'Episode duration value updated from {old_value} to {value}\n'

                    # Learning rate
                    elif command[0] == 'optimizer' and command[1] == 'lr':
                        value = float(command[2])
                        if value < 0 or value > 1:
                            raise ValueError(f'Learning rate value {value} out of range')
                        self.optimizer[2] = 1
                        self.optimizer[3] = value
                        output += f'Learning rate value updated from {self.optimizer[0]} to {value}\n'

                    # Decay
                    elif command[0] == 'optimizer' and command[1] == 'decay':
                        value = float(command[2])
                        if value < 0 or value > 1:
                            raise ValueError(f'Decay value {value} out of range')
                        self.optimizer[4] = 1
                        self.optimizer[5] = value
                        output += f'Decay value updated from {self.optimizer[1]} to {value}\n'

                    # Agent preview
                    elif command[0] == 'preview':
                        camera_values = [0, 0, 0, 0, 0]
                        agent = int(command[1]) if command[1] != 'all' else -1
                        if command[2] == 'on' or command[2] == 'env':
                            value = 1
                        elif command[2] == 'agent':
                            value = 2
                        elif command[2].startswith('cam_'):
                            camera = int(command[2].split('_')[1]) - 1
                            if camera < 0 or camera >= len(settings.PREVIEW_CAMERA_RES):
                                raise Exception('Agent preview camera does not exist')
                            value = camera + 10
                        elif command[2] == 'off':
                            value = 0
                        elif ',' in command[2]:
                            camera_values = [float(value) for value in command[2].split(',')]
                            if len(camera_values) != 5:
                                raise Exception('Agent preview custom camera requires exactly 5 numerical values')
                            value = 3
                        else:
                            raise Exception('Agent preview subcommand invalid')
                        if agent != -1 and (agent < 1 or agent > len(self.agent_show_preview)):
                            raise ValueError(f'Agent number {value} out of range')
                        if agent == -1:
                            for agent in range(len(self.agent_show_preview)):
                                self.agent_show_preview[agent][0] = value
                                self.agent_show_preview[agent][1] = camera_values[0]
                                self.agent_show_preview[agent][2] = camera_values[1]
                                self.agent_show_preview[agent][3] = camera_values[2]
                                self.agent_show_preview[agent][4] = camera_values[3]
                                self.agent_show_preview[agent][5] = camera_values[4]
                            output += f'Preview for all agents toggled {"on" if value else "off"}\n'
                        else:
                            self.agent_show_preview[agent - 1][0] = value
                            self.agent_show_preview[agent - 1][1] = camera_values[0]
                            self.agent_show_preview[agent - 1][2] = camera_values[1]
                            self.agent_show_preview[agent - 1][3] = camera_values[2]
                            self.agent_show_preview[agent - 1][4] = camera_values[3]
                            self.agent_show_preview[agent - 1][5] = camera_values[4]
                            output += f'Preview for agent {agent} toggled {"on" if value else "off"}\n'

                    # Car NPCs
                    elif command[0] == 'carnpcs' and command[1] == 'keep':
                        value = int(command[2])
                        if value < 0 or value > 500:
                            raise ValueError(f'Car NPC number value {value} out of range')
                        old_value = self.car_npcs[0]
                        self.car_npcs[0] = value
                        output += f'Car NPC number value updated from {old_value} to {value}\n'

                    # Car NPCs' reset interval
                    elif command[0] == 'carnpcs' and command[1] == 'reset_interval':
                        value = int(command[2])
                        if value < 0:
                            raise ValueError(f'Car NPC reset interval value {value} out of range')
                        old_value = self.car_npcs[0]
                        self.car_npcs[0] = value
                        output += f'Car NPC reset interval value updated from {old_value} to {value}\n'

                    # Stop training
                    elif command[0] == 'stop':
                        if command[1] == 'now':
                            self.stop.value = STOP.now
                        elif command[1] == 'checkpoint':
                            self.stop.value = STOP.at_checkpoint
                        else:
                            raise Exception(f'Stop subcommand invalid')
                        output += f'Stopping\n'

                    # 'Values' command
                    elif command[0] == 'values':
                        output += '\nCurrent values:\n'
                        output += f'epsilon = {str(list(self.epsilon))}  # [current, decay, min]\n'
                        output += f'discount = {self.discount.value}\n'
                        output += f'update_target_every = {self.update_target_every.value}\n'
                        output += f'min_reward = {self.min_reward.value}\n'
                        output += f'agent_show_preview = {[agent+1 for agent, state in enumerate(self.agent_show_preview) if state.value]}\n'
                        output += f'save_checkpoint_every = {self.save_checkpoint_every.value}\n'
                        output += f'seconds_per_episode = {self.seconds_per_episode.value}\n'
                        output += f'optimizer = [{self.optimizer[0]}, {self.optimizer[1]}]  # [lr, decay]\n'
                        output += f'car_npcs = [{self.car_npcs[0]}, {self.car_npcs[1]}]  # [keep, interval]\n'

                    # Wrong command
                    else:
                        output += f'Command not recognized'

            except Exception as e:
                error += str(e) + '\n'

            # Remove command file
            try:
                os.remove('tmp/' + command_file)
            except Exception as e:
                error += str(e) + '\n'

        # Add error messages to the output, if there are any
        output += error

        # Save response file
        if output:
            with open(f'tmp/output_{int(time.time())}', 'w', encoding='utf-8') as f:
                f.write(output)
