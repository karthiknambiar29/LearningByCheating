from __future__ import print_function
import argparse
import collections
import datetime
import glob
import logging
import math
import os
from threading import local
from numpy import random
import re
import sys
import weakref
from pathlib import Path
import queue
import tqdm
import lmdb
import torch
import time
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_w
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_s
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_SPACE
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append('../../LearningByCheating')
# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import sys
sys.path.append('../PythonAPI/carla')
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.local_planner import RoadOption
from bird_view.models.image import ImageAgent, ImagePolicyModelSS

try:
    sys.path.append('/opt/carla-simulator')
except IndexError:
    pass
from bird_view.utils.map_utils import Wrapper as map_utils
from data_util import YamlConfig, load_config, visualize_birdview, get_actor_blueprints, get_birdview, process, carla_img_to_np, is_within_distance_ahead, find_weather_presets, get_actor_display_name, get_actor_blueprints
from pygame.locals import Color
RED = Color('red')
BLUE = Color('blue')
GREEN = Color('green')
BLACK = Color('black')
COLORS = [RED, BLUE, GREEN, BLACK]
VEHICLE_NAME = 'vehicle.ford.mustang'

PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    2: carla.WeatherParameters.CloudyNoon,
    3: carla.WeatherParameters.WetNoon,
    4: carla.WeatherParameters.WetCloudyNoon,
    5: carla.WeatherParameters.MidRainyNoon,
    6: carla.WeatherParameters.HardRainNoon,
    7: carla.WeatherParameters.SoftRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    9: carla.WeatherParameters.CloudySunset,
    10: carla.WeatherParameters.WetSunset,
    11: carla.WeatherParameters.WetCloudySunset,
    12: carla.WeatherParameters.MidRainSunset,
    13: carla.WeatherParameters.HardRainSunset,
    14: carla.WeatherParameters.SoftRainSunset,
}

class World(object):

    def __init__ (self, world, client, traffic_manager, synchronous_master, args):
        self.args = args
        self.world = world
        self.client = client
        self.map = self.world.get_map()
        self.player = None
        self._actor_filter = VEHICLE_NAME

        # CAMERA PARAMETERS
        self.rgb_queue_left= None
        self.rgb_image_left = None
        self.rgb_queue_right = None
        self.rgb_image_right = None
        self.rgb_queue_bev = None
        self.rgb_image_bev = None
        self.rgb_camera_bp_left = None
        self.rgb_camera_bp_right = None
        self.rgb_camera_bp_bev = None

        #SENSORS
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.total_distance = 0
        self.initial_location = None
        self.current_location = None
        self.frames = 0

        self.observations = {}
        
        # NOTIFICATION SETUP
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)

        self.restart()
        self.generate_traffic(traffic_manager, synchronous_master)

    def restart(self):
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        vehicles = self.world.get_actors().filter('vehicle.*')
        for v in vehicles:
            _ = v.destroy()
        walkers = self.world.get_actors().filter('walker.*')
        for w in walkers:
            w.destroy()

        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # SPAWN THE PLAYER
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            # spawn_points = self.map.get_spawn_points()
            self._start_pose = random.choice(self.map.get_spawn_points())
            self.player = self.world.try_spawn_actor(blueprint, self._start_pose)

        # SPAWN THE CAMERA
        self.rgb_queue_left = queue.Queue()
        rgb_camera_bp_left = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp_left.set_attribute('image_size_x', str(self.args.camera.left.width))
        rgb_camera_bp_left.set_attribute('image_size_y', str(self.args.camera.left.height))
        rgb_camera_bp_left.set_attribute('fov', str(self.args.camera.left.fov))
        self.rgb_camera_bp_left = self.world.spawn_actor(
            rgb_camera_bp_left,
            carla.Transform(carla.Location(x=self.args.camera.left.x, z=self.args.camera.left.z, y=self.args.camera.left.y), carla.Rotation(pitch=self.args.camera.left.pitch, roll=self.args.camera.left.roll, yaw=self.args.camera.left.yaw)),
            attach_to=self.player)
        self.rgb_camera_bp_left.listen(self.rgb_queue_left.put)

        self.rgb_queue_right = queue.Queue()
        rgb_camera_bp_right = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp_right.set_attribute('image_size_x', str(self.args.camera.right.width))
        rgb_camera_bp_right.set_attribute('image_size_y', str(self.args.camera.right.height))
        rgb_camera_bp_right.set_attribute('fov', str(self.args.camera.right.fov))
        self.rgb_camera_bp_right = self.world.spawn_actor(
            rgb_camera_bp_right,
            carla.Transform(carla.Location(x=self.args.camera.right.x, z=self.args.camera.right.z, y=self.args.camera.right.y), carla.Rotation(pitch=self.args.camera.right.pitch, roll=self.args.camera.right.roll, yaw=self.args.camera.right.yaw)),
            attach_to=self.player)
        self.rgb_camera_bp_right.listen(self.rgb_queue_right.put)
        self.rgb_queue_bev = queue.Queue()
        rgb_camera_bp_bev = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp_bev.set_attribute('image_size_x', str(320))
        rgb_camera_bp_bev.set_attribute('image_size_y', str(320))
        rgb_camera_bp_bev.set_attribute('fov', str(self.args.camera.right.fov))
        self.rgb_camera_bp_bev = self.world.spawn_actor(
            rgb_camera_bp_bev,
            carla.Transform(carla.Location(x=0, z=20, y=0), carla.Rotation(pitch=-90, roll=self.args.camera.right.roll, yaw=self.args.camera.right.yaw)),
            attach_to=self.player)
        self.rgb_camera_bp_bev.listen(self.rgb_queue_bev.put)

        #SENSORS
        self.collision_sensor = CollisionSensor(self.player)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player)

        self.initial_location = self._start_pose.location
        # print(self.initial_location)
        # self.total_distance -= self.initial_location.distance(carla.Location())
        # print(self.total_distance)
        if self.args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        
        # map_utils.init(self.client, self.world, self.map, self.player)

    def tick(self):
        """Method for every tick"""
        # self.command = self._local_planner.checkpoint[1]
        # map_utils.tick()
        self.current_location = self.player.get_location()
        self.total_distance += self.current_location.distance(self.initial_location)
        self.initial_location = self.current_location
        while self.rgb_image_left is None or self.rgb_queue_left.qsize() > 0:
            self.rgb_image_left = self.rgb_queue_left.get()
        while self.rgb_image_right is None or self.rgb_queue_right.qsize() > 0:
            self.rgb_image_right = self.rgb_queue_right.get()
        while self.rgb_image_bev is None or self.rgb_queue_bev.qsize() > 0:
            self.rgb_image_bev = self.rgb_queue_bev.get()


    
    def render_image(self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def render(self, display, agent):
        if self.rgb_image_left is not None:
            display.blit(self.render_image(self.rgb_image_left), (0, 0))
        if self.rgb_image_right is not None:
            display.blit(self.render_image(self.rgb_image_right), (0, 160))
        if self.rgb_image_bev is not None:
            display.blit(self.render_image(self.rgb_image_bev), (384, 0))

        self.observations = self.get_observations(agent)
        observations = self.observations
        # observations = self.get_observations(agent)
        # bird_view = visualize_birdview(get_birdview(self.get_observations(agent)))
        # display.blit(pygame.surfarray.make_surface(np.transpose(bird_view, (1, 0, 2))), (384, 0))
        display.blit(pygame.surfarray.make_surface(np.zeros((320, 320))), (704, 0))
        # VELOCITY
        v_offset = 4
        bar_width = 106
        items = list()
        vel = self.player.get_velocity()
        speed = 'Speed:   % 15.3f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
        items.append(str(speed))

        # CONTROL
        control = observations['control']
        items.append(('Throttle:   % 12.3f' % (control.throttle), control.throttle, 1))
        items.append(('Steer:   % 15.3f' % (control.steer), control.steer, -1))
        items.append(('Brake:   % 15.3f' % (control.brake), control.brake, 1))

        # COMMAND
        commands = {
                -1:"VOID",
                1: "LEFT",
                2:"RIGHT",
                3:"STRAIGHT",
                4:"LANEFOLLOW",
                5:"CHANGELANELEFT",
                6:"CHANGELANERIGHT"
        }
        items.append('Command:   % 15s' % (commands[int(agent._local_planner.command.value)]))
        items.append('Traffic:   % 15s' % ('RED' if observations['traffic_light'] == 1.0 else 'GREEN'))
        items.append(f"Total Distance Traveled: {self.total_distance:.2f} meters")
        # items.append(f"Lane Invasion: {self.lane_invasion_sensor.number:.0f}")
        
        # for collision in self.lane_invasion_sensor.laneinvasion_history:
        #     items.append(collision)
        # items.append(f"Collision: {self.collision_sensor.number:.0f}")
        # for collision in self.collision_sensor.collision_history:
        #     items.append(collision)
        # TRAFFIC LIGHT
        # traffic_light = observations['traffic_light']
        # items.append('Traffic Light:   % 12s' % ('RED' if traffic_light == 1.0 else 'GREEN'))
        
        # DISPLAY RENDERING
        for item in items:
            if isinstance(item, tuple):
                if item[-1] < 0:
                    surface = self._font_mono.render(item[0], True, (255, 255, 255))
                    display.blit(surface, (704, v_offset))
                    rect_border = pygame.Rect((900, v_offset + 4), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    rect = pygame.Rect((900 + (item[1]+1)/2 * (bar_width - 6), v_offset + 4), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                    v_offset +=18
                else:    
                    surface = self._font_mono.render(item[0], True, (255, 255, 255))
                    display.blit(surface, (704, v_offset))
                    rect_border = pygame.Rect((900, v_offset + 4), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    rect = pygame.Rect((900, v_offset+4), (item[1] * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                    v_offset +=18
            else:
                surface = self._font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (704, v_offset))
                v_offset +=18
        
        
        pygame.display.update()
    def destroy(self):
        actors = [
            self.player,
            self.rgb_camera_bp_left,
            self.rgb_camera_bp_right,
            self.rgb_camera_bp_bev,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(0.5)

    def get_observations(self, agent):
        result = dict()
        # result.update(map_utils.get_observations())
        pos = self.player.get_location()
        ori = self.player.get_transform().get_forward_vector()
        vel = self.player.get_velocity()
        acc = self.player.get_acceleration()
        traffic_light = 1.0 if agent.traffic_light_manager() else 0.0
        command = agent._local_planner.target_road_option
        control = self.player.get_control()
        result.update({
                'position': np.float32([pos.x, pos.y, pos.z]),
                'orientation': np.float32([ori.x, ori.y]),
                'velocity': np.float32([vel.x, vel.y, vel.z]),
                'acceleration': np.float32([acc.x, acc.y, acc.z]),
                'traffic_light': traffic_light,
                'command': command if command is not None else RoadOption.LANEFOLLOW,
                'control': control})
        # print ("%.3f, %.3f"%(self.rgb_image.timestamp, self._world.get_snapshot().timestamp.elapsed_seconds))
        result.update({
            'rgb_left': carla_img_to_np(self.rgb_image_left),
            'rgb_right': carla_img_to_np(self.rgb_image_right),
            # 'birdview': get_birdview(result),
            # 'collided': self.collided
            })
        if self.args.debug:
            self.frames +=1
            result.update({
                'distance': self.total_distance,
                'frames': self.frames,
                'lane_invasion': self.lane_invasion_sensor.number,
                'collision': self.collision_sensor.number
            })

        self.observations = result
        return result
    
    def generate_traffic(self, traffic_manager, synchronous_master):
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        blueprints = get_actor_blueprints(self.world, self.args.filterv, self.args.generationv)
        blueprintsWalkers = get_actor_blueprints(self.world, self.args.filterw, self.args.generationw)

        if self.args.safe:
            print('safe')
            # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            # blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            # blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            # blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            # blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            # blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)


        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if self.args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.args.number_of_vehicles, number_of_spawn_points)
            self.args.number_of_vehicles = number_of_spawn_points
        
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
            
        for response in self.client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if self.args.car_lights_on:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)
        
        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if self.args.seedw:
            self.world.set_pedestrians_seed(self.args.seedw)
            random.seed(self.args.seedw)

        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id

        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if self.args.sync and  synchronous_master:
            self.world.tick()            
        else:
            self.world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.collision_history = []
        self.number = 0
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        self.number +=1
        actor_type = get_actor_display_name(event.other_actor)
        print('Collision with %r' % actor_type)
        self.collision_history.append('Collision with %r' % actor_type)
        if len(self.collision_history) > 3:
            self.collision_history.pop(0)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self.laneinvasion_history = []
        self.number = 0
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        self.number +=1
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        print('Crossed line %s' % ' and '.join(text))
        self.laneinvasion_history.append('Crossed line %s' % ' and '.join(text))
        if len(self.laneinvasion_history) > 3:
            self.laneinvasion_history.pop(0)

class KeyboardControl(object):
    def __init__(self, world):

        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_light_state(self._lights)
            self._cmd = 4

    def parse_events(self, world):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                
                if isinstance(self._control, carla.VehicleControl):
                    self._parse_vehicle_keys(pygame.key.get_pressed())
                    self._control.reverse = self._control.gear < 0
                    # Set automatic control-related vehicle lights
                    if self._control.brake:
                        current_lights |= carla.VehicleLightState.Brake
                    else: # Remove the Brake flag
                        current_lights &= ~carla.VehicleLightState.Brake
                    if self._control.reverse:
                        current_lights |= carla.VehicleLightState.Reverse
                    else: # Remove the Reverse flag
                        current_lights &= ~carla.VehicleLightState.Reverse
                    if current_lights != self._lights: # Change the light state only if necessary
                        self._lights = current_lights
                        world.player.set_light_state(carla.VehicleLightState(self._lights))
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys):
        if keys[K_UP] or keys[K_w]:
            self._cmd = 4

        if keys[K_DOWN] or keys[K_s]:
            self._cmd = 3

        if keys[K_LEFT] or keys[K_a]:
            self._cmd = 1
        elif keys[K_RIGHT] or keys[K_d]:
            self._cmd = 2
        self._traffic = keys[K_SPACE]


    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    synchronous_master = False
    image_net = ImagePolicyModelSS(backbone='resnet34', all_branch=args.all_branch).to('cuda')
    image_net.load_state_dict(torch.load(args.model_path))
    image_net.eval()

    if args.debug:
        DATA = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        sim_world = client.load_world(args.town)
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
            sim_world.apply_settings(settings)
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        
        world = World(sim_world, client, traffic_manager, synchronous_master, args)
        
        # agent = BehaviorAgent(world.player, behavior=args.behaviour)
        agent = ImageAgent(vehicle=world.player, model=image_net, debug=args.debug)
        # controller = KeyboardControl(world)


        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        clock = pygame.time.Clock()

        # initial_location = world.player.get_location()
        # total_distance = 0
        while True:
            pygame.event.get()
            clock.tick()
            if args.sync:
                sim_world.tick()
            else:
                sim_world.wait_for_tick()

            if args.sync:
                sim_world.tick()
            # if len(FRAMES) % 100 == 0:
            #     np.save('run2/frames.npy', FRAMES)
            # if controller.parse_events(world):
            #     return
            # print(controller._cmd)
            world.tick()
            world.render(display, agent)
            pygame.display.flip()
            if agent.done():
                if args.loop: #args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break
            observations = world.observations
            if args.debug:
                if image_net.all_branch:
                    control, model_preds, world_preds, debug = agent.run_step(observations)
                    for i, model_pred in enumerate(model_preds):
                        for x, y in model_pred:
                            pygame.draw.rect(display, COLORS[i], pygame.Rect(int(x), int(y), 3, 3))
                            pygame.draw.rect(display, COLORS[i], pygame.Rect(int(x), int(y+160), 3, 3))
                else:
                    control, model_pred, world_pred, debug = agent.run_step(observations)
                    for x, y in model_pred:
                        pygame.draw.rect(display, RED, pygame.Rect(int(x), int(y), 3, 3))
                        pygame.draw.rect(display, RED, pygame.Rect(int(x), int(y+160), 3, 3))
            else:
                if image_net.all_branch:
                    control, model_preds, world_preds = agent.run_step(observations)
                    for i, model_pred in enumerate(model_preds):
                        for x, y in model_pred:
                            pygame.draw.rect(display, COLORS[i], pygame.Rect(int(x), int(y), 3, 3))
                            pygame.draw.rect(display, COLORS[i], pygame.Rect(int(x), int(y+160), 3, 3))
                else:
                    control, model_pred, world_pred = agent.run_step(observations)
                    for x, y in model_pred:
                        pygame.draw.rect(display, RED, pygame.Rect(int(x), int(y), 3, 3))
                        pygame.draw.rect(display, RED, pygame.Rect(int(x), int(y+160), 3, 3))

            pygame.display.update()
            if args.debug:
                frames = observations['frames']
                distance = observations['distance']
                lane_invasion = observations['lane_invasion']
                collision = observations['collision']
                acceleration = debug['acceleration']
                throttle = debug['throttle']
                steer = debug['steer']
                alpha = debug['alpha']
                speed = debug['speed']
                target_speed = debug['target_speed']
                command = debug['command']
                traffic = debug['traffic']
                brake = debug['brake']
                DEBUG = np.array([frames, distance, lane_invasion, collision, acceleration, throttle, steer, alpha, speed, target_speed, command, traffic, brake])
                DATA.append(DEBUG)
                pygame.image.save(display, "run7/frame_{}.jpeg".format(frames))
                control.manual_gear_shift = False
            world.player.apply_control(control)

        if original_settings:
            sim_world.apply_settings(original_settings)

    finally:
        np.save('run7/data.npy', np.array(DATA).T)

        if original_settings:
            sim_world.apply_settings(original_settings)
        
        if world is not None:
            world.destroy()
        pygame.quit()



def main():
    args = YamlConfig.from_nested_dicts(load_config('config/hound_config.yaml'))
    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()
