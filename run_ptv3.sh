xhost +local:docker

docker run --runtime=nvidia --gpus all -it --rm --name prism_topomap_semantic_container \
--net=host \
--env="DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--device /dev/nvidia0:/dev/nvidia0 \
--privileged \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e NVIDIA_VISIBLE_DEVICES=all \
-v ${HOME}/TopoSLAM/prism_topomap_semantic_docker/data:/data \
-v ${HOME}/TopoSLAM/prism_topomap_semantic_docker/catkin_ws:/home/docker_prism/catkin_ws \
-v /media:/media \
-v /dev:/dev \
-v /tmp/.Xauthority:/home/docker_prism/.Xauthority:rw \
-v /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
-v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
prism_topomap_semantic_ptv3