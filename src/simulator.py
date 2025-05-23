import carla
import numpy as np
import random
from collections import deque

import torch

class SensorsData:
    def __init__(self, sequence_len=30):
        self.sensors = dict()
        self.id = 0
        self.sequence_len = sequence_len
        self.current_load = dict()
    
    def add_sensor(self, sensor_type:str = 'camera', direction:str = 'front'):
        self.sensors[f'{sensor_type}_{direction}'] = deque(maxlen=self.sequence_len * 30)
        self.current_load[f'{sensor_type}_{direction}'] = 0
    
    def update_sensor_data(self, data, sensor_name:str = 'camera_front'):
        if sensor_name.find('camera') != -1:
            # if self.current_load[sensor_name] == self.sequence_len:
            # self.sensors[sensor_name].popleft()
                # del to_delete
            self.sensors[sensor_name].append(torch.from_numpy(data).permute(0, 3, 1, 2))
        else:
            self.sensors[sensor_name].append(data)
        # print(self.sensors[sensor_name].shape)

    def get_sensor_data(self, sensor_name:str = 'camera_front'):
        return self.sensors[sensor_name]

class Simulator:
    def __init__(self,
                 world_name : str='Town01',
                 debug=True,
                 dump_data=True,
                 sequence_len=30) -> None:
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10)
        self.world_name = world_name
        self.world = self.client.get_world()
        self.client.load_world(world_name)
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.dump_data = dump_data
        self.image_frame = None
        
        self.sequence_len = sequence_len
        self.sensors_data = SensorsData(sequence_len=self.sequence_len)
        
        if debug:
            self.world.unload_map_layer(carla.MapLayer.Buildings)
            self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
            self.world.unload_map_layer(carla.MapLayer.StreetLights)
    
    def spawn_camera(self, 
                     rel_coordinates : carla.Location = carla.Location(x=2, z=1.5),
                     fov = 90,
                     image_param = (224, 224)):
        self.image_param = image_param
        cam_rel_pos = carla.Transform(rel_coordinates)

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb') # actually rgba
        camera_bp.set_attribute('image_size_x', str(image_param[0]))
        camera_bp.set_attribute('image_size_y', str(image_param[1]))
        camera_bp.set_attribute('fov', str(fov))
        
        self.camera = self.world.try_spawn_actor(camera_bp, cam_rel_pos, attach_to=self.vehicle)
        self.sensors_data.add_sensor(sensor_type='camera', direction='front')
    
    def spawn_collision_sensor(self,
                               rel_coordinates : carla.Location = carla.Location(x=2.7, z=1)):
        col_sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        col_sensor_rel_pos = carla.Transform(rel_coordinates)
        self.collision_sensor = self.world.try_spawn_actor(col_sensor_bp, col_sensor_rel_pos, attach_to=self.vehicle)
        self.sensors_data.add_sensor(sensor_type='collision', direction='front')

    def spawn_vehicle(self,
                      model : str = 'vehicle.nissan.patrol_2021',
                      coordinates : carla.Location = carla.Location(0,0,0),
                      rotation : carla.Rotation = carla.Rotation(0,0,0)): 
        vehicle_bp = self.world.get_blueprint_library().find(model)
        if not self.vehicle:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, carla.Transform(coordinates, rotation))
        self.spawn_collision_sensor()

    def spawn_car_with_camera(self,
                              model : str = 'vehicle.nissan.patrol_2021',
                              vehicle_coordinates=carla.Location(0,0,0),
                              vehicle_rotation=carla.Rotation(0,0,0),
                              rel_coordinates : carla.Location = carla.Location(x=2, z=1.5),
                              fov = 90, image_param = (224, 224)):
        self.spawn_vehicle(model, vehicle_coordinates, vehicle_rotation)
        self.spawn_camera(rel_coordinates, fov, image_param)

    def get_vehicle(self):
        return self.vehicle
    
    def get_camera(self):
        return self.camera
    
    def default_camera_callback(self, image):
        def get_image_data(image):
            self.image_frame = image.frame
            if self.dump_data is True:
                image.save_to_disk(f"out/{self.world_name}/{image.frame}.png")
            img = np.array(image.raw_data)
            img = img.reshape((1, self.image_param[0], self.image_param[1], 4))
            img = img[:, :, :, :3] #drop alpha

            return (img / 255.0).copy() # normalize data
        
        data = get_image_data(image)
        self.sensors_data.update_sensor_data(data, 'camera_front')
        # print(self.sensors_data.get_sensor_data(sensor_name='camera_front'))
        return data

    def start_camera_stream(self, callback = None):
        if self.camera:
            if callback is None:
                callback = self.default_camera_callback
            self.camera.listen(lambda image: callback(image))
            print(f'Camera stream started')

    def stop_camera_stream(self):
        if self.camera:
            self.camera.stop()
            print(f'Camera stream stopped')

    def start_collision_sensor(self, sensor_name = 'collision_front'):
        self.collision_sensor.listen(
            lambda data: self.sensors_data.update_sensor_data(data, sensor_name))
    
    def stop_collision_sensor(self, sensor_name = 'collision_front'):
        self.collision_sensor.stop()

    
    def destroy_all(self):
        self.stop_camera_stream()
        self.camera.destroy()
        self.stop_collision_sensor()
        self.collision_sensor.destroy()
        self.vehicle.destroy()
        
        self.collision_sensor = None
        self.camera = None
        self.vehicle = None
    
