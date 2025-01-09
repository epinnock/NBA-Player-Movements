import pandas as pd
import os
import gc
import json
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from Game import Game
from Constant import Constant
from Moment import Moment

def moment_to_json(moment, event_id, moment_id, game_id, image_path):
    json_data = {}
    timestamp = {}
    timestamp['quarter'] = moment[0]
    timestamp['gameclock'] = moment[2]
    timestamp['shotclock'] = moment[3]
    json_data['timestamp'] = timestamp
    json_data['raw_moment'] = moment
    json_data['moment_id'] = moment_id
    json_data['event_id'] = event_id
    json_data['game_id'] = game_id
    json_data['image_path'] = image_path
    return json_data

def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + float(seconds)
    return total_seconds

def file_exists(filename):
    try:
        with open(filename, 'r') as f:
            return True
    except FileNotFoundError:
        return False

def file_in_array(filename, files):
    return filename in files

def get_files(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def render_moment(args):
    print('calling render')
    game_screenshot_gen, moment, event_id, moment_id, game_id, files = args
    folder_path = '/home/boxuser/workspace/nbadataset/frames/'
    image_filename = f'g-{game_id}-e-{event_id}-m-{moment_id}.png'
    image_path=folder_path+image_filename
    if True:
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(Constant.X_MIN, Constant.X_MAX)
            ax.set_ylim(Constant.Y_MIN, Constant.Y_MAX)
            ax.axis('off')
            
            court = plt.imread("/home/boxuser/workspace/nbadataset/NBA-Player-Movements/court.png")
            ax.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF, Constant.Y_MAX, Constant.Y_MIN])
            _moment = moment
            moment = Moment(moment)
           
            for player in moment.players:
                circle = Circle((player.x, player.y), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                ax.add_patch(circle)
                ax.annotate(game_screenshot_gen.event.player_ids_dict[player.id][1], (player.x, player.y), color='w', weight='bold', 
                            fontsize=8, ha='center', va='center')
            
            ball = Circle((moment.ball.x, moment.ball.y), moment.ball.radius / Constant.NORMALIZATION_COEF, color=moment.ball.color)
            ax.add_patch(ball)
            
            clock_info = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                        moment.quarter,
                        int(moment.game_clock) % 3600 // 60,
                        int(moment.game_clock) % 60,
                        moment.shot_clock)
            ax.text(Constant.X_CENTER, Constant.Y_CENTER, clock_info, color='black', ha='center', va='center')
            print(f'saving to file {image_path}\n')
            plt.savefig(image_path, bbox_inches='tight')
        except Exception as e:
            print(f'Error: {e}')
            image_path = "Error"
        finally:
            plt.close(fig)
    
        return moment_to_json(_moment, event_id, moment_id, game_id, image_path)
    else:
        return None

def process_event(event, event_id, game_id, files, path_to_json):
    game_screenshot_gen = Game(path_to_json=path_to_json, event_index=event_id)
    game_screenshot_gen.read_json()
    game_screenshot_gen.set_event(event_id)
    args_list = [(game_screenshot_gen, moment, event_id, moment_id, game_id, files) 
                 for moment_id, moment in enumerate(event['moments'])]
    
    return args_list

def write_batch_to_json(batch_data, file_path):
    # Filter out None entries
    filtered_batch_data = [entry for entry in batch_data if entry is not None]
    
    # Sort the batch data by moment_id
    sorted_batch_data = sorted(filtered_batch_data, key=lambda x: x['moment_id'])
    
    with open(file_path, 'a') as json_file:
        for entry in sorted_batch_data:
            json.dump(entry, json_file)
            json_file.write('\n')

def main():
    path_to_json = '/home/boxuser/workspace/nbadataset/0021500439.json'
    game_df = pd.read_json(path_to_json)
    game_id = '0021500439'
    folder_path = '/home/boxuser/workspace/nbadataset/frames/'
    files = get_files(folder_path)
    output_json_path = '/home/boxuser/workspace/nbadataset/all_game_moments.json'
    batch_size = 1
    event_args_list = []

 # Clear or create the output file
    for start in range(78, len(game_df.events), batch_size):
        end = min(start + batch_size, len(game_df.events))
        batch_events = game_df.events[start:end]
        print(print(batch_events))
        for event_id, event in enumerate(batch_events, start=start):
            event_args_list.extend(process_event(event, event_id, game_id, files, path_to_json))
        print(len(event_args_list))
        with multiprocessing.Pool(processes=48) as pool:
            results = pool.map(render_moment, event_args_list)

            write_batch_to_json(results, output_json_path)
        event_args_list.clear()  # Clear the list to free up memory
        gc.collect()  # Trigger garbage collection

   

if __name__ == "__main__":
    main()