import sys
sys.path.append("CARLA_SIM/PythonAPI/carla/")
import carla
from agents.navigation.basic_agent import BasicAgent
import torch

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

    def run_step(self, dump_data=False):
        raw_data = self.sensors_data_storage.get_sensor_data('camera_front')
        if raw_data is None or len(raw_data) < 10:
            control = super().run_step()
            return control, [0,0,0,0], None
        if dump_data:
            true_control = super().run_step()
            return true_control, None, raw_data
        #print(raw_data)
        model_control, _ = self.model(list(raw_data)[-10:])
        #print(data)
        true_control = super().run_step()

        return true_control, model_control, raw_data

