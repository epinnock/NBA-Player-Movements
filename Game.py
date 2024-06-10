import pandas as pd
from Event import Event
from Team import Team
from Constant import Constant


class Game:
    """A class for keeping info about the games"""
    def __init__(self, path_to_json, event_index):
        # self.events = None
        self.home_team = None
        self.guest_team = None
        self.event = None
        self.path_to_json = path_to_json
        self.event_index = event_index
        self.data_frame = None

    def read_json(self):
        self.data_frame = pd.read_json(self.path_to_json)
        last_default_index = len(self.data_frame) - 1
        self.set_event(last_default_index)
        


    
    def set_event(self,event_id):
      self.event_index = event_id
      index = self.event_index
      event = self.data_frame['events'][index]
      self.event = Event(event)
      self.home_team = Team(event['home']['teamid'])
      self.guest_team = Team(event['visitor']['teamid'])
    print(Constant.MESSAGE + str(last_default_index))

    def start(self):
        self.event.show()
