<<<<<<< HEAD
hhhello
=======
# Docker 
- docker run -p 6080:80 -p 5900:5900 --gpus '"device=1"' --name=lbc -e USER=carla -e PASSWORD=carla --net=bridge --runtime=nvidia -e VNC_PASSWORD=carla -d -v /dev/shm:/dev/shm dorowu/ubuntu-desktop-lxde-vnc:bionic
- docker run --privileged --gpus '"device=1"' --name carla -d --net=bridge -e DISPLAY=$DISPLAY carlasim/carla:0.9.13 /bin/bash ./CarlaUE4.sh -RenderOffScreen
>>>>>>> lbc
