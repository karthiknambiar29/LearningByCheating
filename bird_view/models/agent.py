import numpy as np
import torch
import torchvision.transforms as transforms

import carla
import sys
import glob
import os


sys.path.append(glob.glob('../../LearningByCheating/PythonAPI/carla')[0])

from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.basic_agent import BasicAgent

class Agent(BasicAgent):
    def __init__(self, vehicle, model=None, opt_dict={}, **kwargs):
        assert model is not None

        if len(kwargs) > 0:
            print('Unused kwargs: %s' % kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.ToTensor()

        self.one_hot = torch.FloatTensor(torch.eye(4))

        self.model = model.to(self.device)
        self.model.eval()

        self.debug = dict()
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._sampling_resolution = 2.0

        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict)
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        self._last_traffic_light = None
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        # self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._max_brake = 0.5

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()
    
    def postprocess(self, steer, throttle, brake):
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False

        return control
    
    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    # def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
    #     """
    #     Method to check if there is a red light affecting the vehicle.

    #         :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
    #             If None, all traffic lights in the scene are used
    #         :param max_distance (float): max distance for traffic lights to be considered relevant.
    #             If None, the base threshold value is used
    #     """
    #     if self._ignore_traffic_lights:
    #         return (False, None)

    #     if not lights_list:
    #         lights_list = self._world.get_actors().filter("*traffic_light*")

    #     if not max_distance:
    #         max_distance = self._base_tlight_threshold

    #     if self._last_traffic_light:
    #         if self._last_traffic_light.state != carla.TrafficLightState.Red:
    #             self._last_traffic_light = None
    #         else:
    #             return (True, self._last_traffic_light)

    #     ego_vehicle_location = self._vehicle.get_location()
    #     ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    #     for traffic_light in lights_list:
    #         object_location = get_trafficlight_trigger_location(traffic_light)
    #         object_waypoint = self._map.get_waypoint(object_location)

    #         if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
    #             continue

    #         ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
    #         wp_dir = object_waypoint.transform.get_forward_vector()
    #         dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

    #         if dot_ve_wp < 0:
    #             continue

    #         if traffic_light.state != carla.TrafficLightState.Red:
    #             continue

    #         if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
    #             self._last_traffic_light = traffic_light
    #             return (True, traffic_light)

    #     return (False, None)