#!/bin/bash

# config the screen resolution

#export WANDB_NAME=LUNAR_sweep_$vnc_port+_$jup_port

sed -i "s/1920x1080/1920x1080/" /usr/local/bin/xvfb.sh

# some configurations for LXDE
# mkdir -p /root/.config/pcmanfm/LXDE/
# ln -sf /usr/local/share/doro-lxde-wallpapers/desktop-items-0.conf /root/.config/pcmanfm/LXDE/

# if [ ! -f /catkin_ws/.initialized ]; then
#   cp -r /catkin_ws_initial/* /catkin_ws/
#   touch /catkin_ws/.initialized
#   rm -rf /catkin_ws_initial
# fi

# if [ ! -f /data/.initialized ]; then
#   cp -r /data_initial/* /data/
#   touch /data/.initialized
#   rm -rf /data_initial
# fi

# start all the services
# exec /bin/tini -- /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf

source /opt/ros/noetic/setup.bash

exec /bin/bash