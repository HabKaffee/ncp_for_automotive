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

        self.model = model

    def run_step(self, current_pos: carla.Location, waypoint:carla.Location):
        raw_data = self.sensors_data_storage.get_sensor_data('camera_front')
        data = self.model.encoder.extract_features(raw_data)
        out = self.model.rnn(data)
        collided = False
        print(out[0][0])
        movement = torch.mean(out[0])
        # control = carla.VehicleControl(throttle=0.2, brake=0.0, steer = 0.0)
        direction_vector = np.array([current_pos.x - self.previous_pos.x,
                                    current_pos.y - self.previous_pos.y])
                                    # current_pos.z - self.previous_pos.z])
        vector_to_waypoint = np.array([waypoint.x - current_pos.x,
                                       waypoint.y - current_pos.y])
                                    #    waypoint.z - current_pos.z])
        # print(vector_to_waypoint, direction_vector)
        self.previous_pos = current_pos
        direction_vector = torch.from_numpy(direction_vector)
        vector_to_waypoint = torch.from_numpy(vector_to_waypoint)
        angle_cos = torch.dot(direction_vector, vector_to_waypoint)
        if torch.linalg.vector_norm(direction_vector):
            angle_cos /= torch.linalg.vector_norm(direction_vector)
        if torch.linalg.vector_norm(vector_to_waypoint):
            angle_cos /= torch.linalg.vector_norm(vector_to_waypoint)

        print(f'Angle_cos = {angle_cos}')
        true_angle = torch.acos(angle_cos)
        print(f'Angle in rad = {true_angle}')
        true_angle = true_angle * 180 / np.pi
        print(f'Angle in deg = {true_angle}')
        if true_angle > 90:
            true_angle -= 90
            true_angle *= -1
        true_angle = (true_angle) / 70 # steering angle -70 to 70
        print(f'Steering angle = {true_angle}')
        # print(movement, true_angle)

        if self.sensors_data_storage.get_sensor_data('collision_front') is not None:
            collided = True
        return None, collided, true_angle, raw_data
        #return control, collided, true_angle, raw_data



