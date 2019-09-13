import settings
from sources import TRAINER_STATE_MESSAGE, AGENT_STATE_MESSAGE, STOP, STOP_MESSAGE, CarlaEnv, CARLA_SETTINGS_STATE, CARLA_SETTINGS_STATE_MESSAGE
import time
import colorama


colorama.init()


class ConsoleStats:

    def __init__(self, stop, duration, start_time, episode, epsilon, trainer_stats, agent_stats, episode_stats, carla_fps, weights_iteration, optimizer, carla_settings_threads, seconds_per_episode):
        self.stop = stop
        self.duration = duration
        self.start_time = start_time
        self.episode = episode
        self.epsilon = epsilon
        self.trainer_stats = trainer_stats
        self.agent_stats = agent_stats
        self.episode_stats = episode_stats
        self.carla_fps = carla_fps
        self.weights_iteration = weights_iteration
        self.optimizer = optimizer
        self.carla_settings_threads = carla_settings_threads
        self.seconds_per_episode = seconds_per_episode

    def get_color(self, value=None, type=None):
        if type == 'stop':
            if value in [0, 4]:
                return colorama.Fore.GREEN
            elif value in [1, 2, 3]:
                return colorama.Fore.YELLOW
            elif value in [5, 6, 7]:
                return colorama.Fore.RED
        elif type == 'agent':
            if value in [1, 3]:
                return colorama.Fore.GREEN
            elif value in [0, 2, 5]:
                return colorama.Fore.YELLOW
            elif value == 4:
                return colorama.Fore.RED
        elif type == 'trainer':
            if value in [2, 3]:
                return colorama.Fore.GREEN
            elif value in [0, 1, 4]:
                return colorama.Fore.YELLOW
        elif type == 'settings':
            if value in [1, 3]:
                return colorama.Fore.GREEN
            elif value in [0, 2]:
                return colorama.Fore.YELLOW
            elif value == 4:
                return colorama.Fore.RED
        elif type == 'fps':
            if value >= 15:
                return colorama.Fore.GREEN
            elif value >= 12:
                return colorama.Fore.YELLOW
            else:
                return colorama.Fore.RED
        elif type == 'epsilon':
            if value == self.epsilon[2]:
                return colorama.Fore.GREEN
            elif value <= 0.5:
                return colorama.Fore.YELLOW
            else:
                return colorama.Fore.RED
        elif type == 'time':
            if value >= self.seconds_per_episode.value * 2 / 3:
                return colorama.Fore.GREEN
            elif value >= self.seconds_per_episode.value / 3:
                return colorama.Fore.YELLOW
            else:
                return colorama.Fore.RED
        elif type == 'zero':
            if value > 0:
                return colorama.Fore.GREEN
            elif value == 0:
                return colorama.Fore.YELLOW
            else:
                return colorama.Fore.RED

        return colorama.Fore.RESET

    def print(self):
        while True:

            lines = []

            # 10 empty lines
            for _ in range(10):
                lines.append('')

            # Overall status
            lines.append(f'STATUS: {self.get_color(self.stop.value, "stop")}{STOP_MESSAGE[self.stop.value]}{self.get_color()}')

            # Episode, duration, epsilon
            # Number of seconds from the very beginning
            self.duration.value = time.time() - self.start_time
            duration_string = time.strftime("%H:%M:%S", time.gmtime(self.duration.value))
            # Hacky way of getting number of days - we get numeric day of year and substract 1 as it starts from 1
            duration_days = int(time.strftime("%j", time.gmtime(self.duration.value))) - 1
            # If it's more than a day - add day string
            if duration_days:
                duration_string = f'{duration_days}d ' + duration_string
            lines.append(f'EPISODE: {self.episode.value:6d} | Duration: {duration_string} | Epsilon: {self.get_color(self.epsilon[0], "epsilon")}{self.epsilon[0]:>4.3f}{self.get_color()}')

            # Optimizer
            lines.append(f'OPTIMIZER: Learning rate: {self.optimizer[0]:.5f} | Decay: {self.optimizer[1]:.5f}')

            if settings.SHOW_CARLA_ENV_SETTINGS:
                for process_no in range(settings.CARLA_HOSTS_NO):
                    # Empty line
                    lines.append('')

                    lines.append(f'Carla #{process_no+1}')
                    # Weather and NPCs
                    azimuth = f'{self.carla_settings_threads[process_no][1].weather.sun.azimuth:>7.2f}' if self.carla_settings_threads[process_no][1].weather else '    ---'
                    altitude = f'{self.carla_settings_threads[process_no][1].weather.sun.altitude:>7.2f}' if self.carla_settings_threads[process_no][1].weather else '    ---'
                    clouds = f'{self.carla_settings_threads[process_no][1].weather.storm.clouds:>6.2f}%' if self.carla_settings_threads[process_no][1].weather else '    ---'
                    wind = f'{self.carla_settings_threads[process_no][1].weather.storm.wind:>6.2f}%' if self.carla_settings_threads[process_no][1].weather else '    ---'
                    rain = f'{self.carla_settings_threads[process_no][1].weather.storm.rain:>6.2f}%' if self.carla_settings_threads[process_no][1].weather else '    ---'
                    lines.append(f'Sun     |   AZ: {azimuth} |  ALT: {altitude}')
                    lines.append(f'Storm   | CLDS: {clouds} | RAIN: {rain} | WIND: {wind}')
                    lines.append(f'NPCs: {len(self.carla_settings_threads[process_no][1].spawned_car_npcs) if self.carla_settings_threads[process_no][1].state == CARLA_SETTINGS_STATE.working else "---"}')

            # Empty line
            lines.append('')

            # Reward and episode time
            avg_raw_min_reward = f'{self.get_color(self.episode_stats[0], "zero")}{self.episode_stats[0]:>7.1f}{self.get_color()}' if self.episode_stats[0] != -10**6 else '    ---'
            avg_raw_avg_reward = f'{self.get_color(self.episode_stats[1], "zero")}{self.episode_stats[1]:>7.1f}{self.get_color()}' if self.episode_stats[1] != -10**6 else '    ---'
            avg_raw_max_reward = f'{self.get_color(self.episode_stats[2], "zero")}{self.episode_stats[2]:>7.1f}{self.get_color()}' if self.episode_stats[2] != -10**6 else '    ---'
            avg_weighted_min_reward = f'{self.get_color(self.episode_stats[7], "zero")}{self.episode_stats[7]:>7.1f}{self.get_color()}' if self.episode_stats[0] != -10**6 else '    ---'
            avg_weighted_avg_reward = f'{self.get_color(self.episode_stats[8], "zero")}{self.episode_stats[8]:>7.1f}{self.get_color()}' if self.episode_stats[1] != -10**6 else '    ---'
            avg_weighted_max_reward = f'{self.get_color(self.episode_stats[9], "zero")}{self.episode_stats[9]:>7.1f}{self.get_color()}' if self.episode_stats[2] != -10**6 else '    ---'
            min_episode_time = f'{self.get_color(self.episode_stats[3], "time")}{self.episode_stats[3]:>7.1f}{self.get_color()}' if self.episode_stats[3] > 0 else '    ---'
            avg_episode_time = f'{self.get_color(self.episode_stats[4], "time")}{self.episode_stats[4]:>7.1f}{self.get_color()}' if self.episode_stats[4] > 0 else '    ---'
            max_episode_time = f'{self.get_color(self.episode_stats[5], "time")}{self.episode_stats[5]:>7.1f}{self.get_color()}' if self.episode_stats[5] > 0 else '    ---'
            lines.append(f'Rew.raw | MIN: {avg_raw_min_reward} | AVG: {avg_raw_avg_reward} | MAX: {avg_raw_max_reward}')
            lines.append(f'Rew.wgh | MIN: {avg_weighted_min_reward} | AVG: {avg_weighted_avg_reward} | MAX: {avg_weighted_max_reward}')
            lines.append(f'Ep.time | MIN: {min_episode_time} | AVG: {avg_episode_time} | MAX: {max_episode_time}')

            # Predicted Q values
            line = 'Avg Qs '
            for i in range(1, CarlaEnv.action_space_size + 1):
                line += f' | {i - 1:>3d}: ' + (f'{self.get_color(self.episode_stats[10 + i*3], "zero")}{self.episode_stats[10 + i*3]:>7.1f}{self.get_color()}' if self.episode_stats[10 + i*3] != -10**6 else '    ---')
            lines.append(line)
            line = 'Std Qs '
            for i in range(1, CarlaEnv.action_space_size + 1):
                line += f' | {i - 1:>3d}: ' + (f'{self.episode_stats[11 + i*3]:>7.1f}' if self.episode_stats[11 + i*3] != -10**6 else '    ---')
            lines.append(line)
            line = 'Act dis'
            for i in range(1, CarlaEnv.action_space_size + 1):
                line += f' | {i - 1:>3d}: ' + (f'{self.episode_stats[12 + i*3]:>6.1f}%' if self.episode_stats[12 + i*3] != -10**6 else '    ---')
            lines.append(line)

            # Empty line
            lines.append('')

            # Carla
            for process_no in range(settings.CARLA_HOSTS_NO):
                status = f'{self.get_color(self.carla_settings_threads[process_no][1].state, "settings")}{CARLA_SETTINGS_STATE_MESSAGE[self.carla_settings_threads[process_no][1].state]:>9s}{self.get_color()}'
                fps = self.carla_fps[process_no].value
                world = self.carla_settings_threads[process_no][1].world_name if self.carla_settings_threads[process_no][1].world_name else '---'
                carla_fps = f'{self.get_color(fps, "fps")}{fps:>4.1f}{self.get_color()}' if fps > 0 else ' ---'
                lines.append(f'Carla {process_no+1} | FPS: {carla_fps} | Status: {status} | World: {world}')

            # Trainer TPS
            status = f'{self.get_color(self.trainer_stats[0], "trainer")}{TRAINER_STATE_MESSAGE[self.trainer_stats[0]]:>9s}{self.get_color()}'
            tps = f'{self.trainer_stats[1]:>4.1f}' if self.trainer_stats[1] > 0 else ' ---'
            lines.append(f'Trainer | TPS: {tps} | Status: {status} | Iter: {self.weights_iteration.value:>5d}')

            # Agents FPS
            for agent in range(settings.AGENTS):
                status = f'{self.get_color(self.agent_stats[agent][0], "agent")}{AGENT_STATE_MESSAGE[self.agent_stats[agent][0]]:>9s}{self.get_color()}'
                fps = f'{self.get_color(self.agent_stats[agent][2], "fps")}{self.agent_stats[agent][2]:>4.1f}{self.get_color()}' if self.agent_stats[agent][2] > 0 else ' ---'
                step = f'{self.agent_stats[agent][1]:>5.0f}' if self.agent_stats[agent][1] > 0 else '  ---'
                lines.append(f'Agent {agent+1} | FPS: {fps} | Status: {status} | Step: {step}')

            # Average agent FPS
            avg_fps = f'{self.get_color(self.episode_stats[6], "fps")}{self.episode_stats[6]:>4.1f}{self.get_color()}' if self.episode_stats[6] > 0 else ' ---'
            lines.append(f'Average | FPS: {avg_fps}')

            # Empty line
            lines.append('')

            # Sets cursor back at the beginning of text block (moves it up)
            string = '\r' + ('\033[A' * len(lines))

            # Add spaces to form a 70-char long line
            for line in lines:
                string += line + ' '*(70 - len(line.replace(colorama.Fore.GREEN, '').replace(colorama.Fore.YELLOW, '').replace(colorama.Fore.RED, '').replace(colorama.Fore.RESET, ''))) + '\n'

            print(string, end='')

            if self.stop.value == STOP.stopped:
                return

            time.sleep(0.2)

    @staticmethod
    def print_short(fps_counter, env, qs, action, action_name):
        q_string = []
        for q in qs:
            q_string.append(f'{q:>5.2f}')
        q_string = ', '.join(q_string)
        print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Carla: {len(env.frametimes)/sum(env.frametimes):>4.1f} FPS | Action: [{q_string}] {action} {action_name}')
