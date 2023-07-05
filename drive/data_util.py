import numpy as np
from carla import ColorConverter as cc
import carla
import math
import re
import yaml
def load_config(file_path):
    with open(file_path, "r") as config_file:
        config_dict = yaml.safe_load(config_file)
        return config_dict

class YamlConfig(dict):
        def __init__(self, *args, **kwargs):
            super(YamlConfig, self).__init__(*args, **kwargs)
            self.__dict__ = self
    
        @classmethod
        def from_nested_dicts(cls, data):
            """ Construct nested AttrDicts from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return cls({key: cls.from_nested_dicts(data[key]) for key in data})

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

BACKGROUND = [0, 47, 0]
COLORS = [
        (102, 102, 102),
        (253, 253, 17),
        (255, 255, 255),
        (204, 6, 5),
        (250, 210, 1),
        (39, 232, 51),
        (0, 0, 142),
        (220, 20, 60)
        ]
def visualize_birdview(birdview):
    """
    0 road
    1 lane
    2 red light
    3 yellow light
    4 green light
    5 vehicle
    6 pedestrian
    """
    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(len(COLORS)):
        canvas[birdview[:,:,i] > 0] = COLORS[i]

    return canvas

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
def get_birdview(observations):
    birdview = [
            observations['road'],
            observations['lane'],
            observations['hero'],
            observations['traffic'],
            observations['vehicle'],
            observations['pedestrian']
            ]
    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview

def process(observations):
    result = dict()
    result['rgb_left'] = observations['rgb_left'].copy()
    result['rgb_right'] = observations['rgb_right'].copy()
    result['birdview'] = observations['birdview'].copy()
    control = observations['control']
    control = [control.steer, control.throttle, control.brake]

    result['control'] = np.float32(control)
    measurements = [
        observations['position'],
        observations['orientation'],
        observations['velocity'],
        observations['acceleration'],
        observations['command'].value,
        observations['control'].steer,
        observations['control'].throttle,
        observations['control'].brake,
        observations['control'].manual_gear_shift,
        observations['control'].gear,
        observations['traffic_light']
        ]
    measurements = [x if isinstance(x, np.ndarray) else np.float32([x]) for x in measurements]
    measurements = np.concatenate(measurements, 0)

    result['measurements'] = measurements

    return result

def carla_img_to_np(carla_img):
    carla_img.convert(cc.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:,:,:3]
    img = img[:,:,::-1]

    return img

def is_within_distance_ahead(target_location, current_location, orientation, max_distance, degree=60):
    u = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / distance))

    return angle < degree

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name