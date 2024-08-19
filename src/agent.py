import sys
sys.path.append("CARLA_SIM/PythonAPI/carla/")
import carla
from agents.navigation.basic_agent import BasicAgent
import torch
from random import randint
import numpy as np

from src.simulator import Simulator

from src.model import Model

class NCPAgent(BasicAgent):
    def __init__(self,
                 simulator : Simulator,
                 model : Model,
                 target_speed=20):
        super().__init__(simulator.vehicle, target_speed)
        self.simulator = simulator
        self.vehicle = self.simulator.vehicle
        self.camera = self.simulator.camera
        self.collision_sensor = self.simulator.collision_sensor
        self.sensors_data_storage = self.simulator.sensors_data

        self.simulator.start_camera_stream()
        self.simulator.start_collision_sensor()
        self.previous_pos = self.vehicle.get_location()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model

    def run_step(self):
        raw_data = self.sensors_data_storage.get_sensor_data('camera_front')
        if raw_data is None:
            control = super().run_step()
            return control, 0, None, 0
        data = self.model.extract_features(raw_data)
        data = data.to(self.device)
        out = self.model.rnn(data)
        movement = torch.mean(out[0])
        control = super().run_step()

        return control, movement, raw_data, out[0]

