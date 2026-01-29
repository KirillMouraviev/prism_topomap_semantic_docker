# Запуск PRISM-TopoMap в докере:

## Настройка среды и создание контейнера

1. Склонируйте этот репозиторий и прокиньте все подмодули (PRISM-TopoMap, habitat_ros, toposlam_msgs):

`git clone https://github.com/KirillMouraviev/prism_topomap_semantic_docker`

`cd prism_topomap_semantic_docker`

`git submodule update --init --recursive`

2. Создайте папки под датасеты, карты и т.д.:

`cd data`

`mkdir -p maps/prism_topomap`

`mkdir scene_datasets`

`mkdir logs`

`mkdir weights`

3. Загрузите данные и веса моделей:
- https://disk.yandex.ru/d/_dJPBG213Wc5CA - в папку `data/scene_datasets`
- https://disk.yandex.ru/d/SPd4sqqxyOo8Pg - в папку `data`
- https://disk.yandex.ru/d/Hu9rK92tAZ3GdA - в папку `data/weights`
- https://disk.yandex.ru/d/Z5pHeq-1iSYeQQ - в папку `data/models`

## Запуск контейнера и PRISM-TopoMap:

1. Запустите контейнер через Docker Compose. Перед этим не забудьте прокинуть дисплей:

`xhost +local:`

`docker compose run prism_topomap_habitat`

В контейнер прокидываются директории `catkin_ws` (в `/home/docker_prism/catkin_ws`) и `data` (в `/data`).

2. В контейнере соберите ROS Workspace:

`cd catkin_ws`

`catkin_make`

3. Дальше в контейнере удобнее всего будет работать в tmux, создав несколько терминалов

4. В первом терминале запустите `roscore`

5. Во втором терминале запустите симулятор Habitat:

`sudo su` (для библиотеки keyboard (управление агентом с клавиатуры) нужен root)

`source devel/setup.bash`

`roslaunch habitat_ros toposlam_experiment_mp3d_4x90.launch scene_name:=<scene_name> agent_type:=<agent_type>`

Доступные варианты `agent_type`:
- `keyboard` - управление стрелками на клавиатуре
- `shortest_path_follower` - объезд среды по кратчайшим путям между точками, записанными в файлах из папки `catkin_ws/src/habitat_ros/goal_positions`
- `greedy_path_follower` - следование пути, который приходит через ROS
- `ddppo` - модель DD-PPO, которая получает данные (картинку и pointgoal) через ROS
Доступные варианты `scene_name` можно посмотреть в папке `data/scene_datasets/mp3d_toposlam_validation_scenes`

6. В третьем терминале запустите PRISM-TopoMap:

`source devel/setup.bash`

`roslaunch prism_topomap build_map_by_iou_habitat.launch`

После завершения он сохранит карту в папку `data/maps/prism_topomap/<scene_name>`

7. Для просмотра карты, которая строится методом PRISM-TopoMap, можно открыть готовый конфиг RViz в четвертом терминале:

`cd src/PRISM-TopoMap/rviz`

`rviz -d toposlam_experiment_habitat.rviz`

## Запуск PRISM-TopoMap на бэге с реального робота в контейнере
1. Выполните шаги 1-4 из предыдущего раздела

2. Создайте папку `data/bags` и скачайте rosbag туда (ссылка: https://disk.yandex.ru/d/BT2iuAQKsmswmg)

3. Во втором терминале запустите PRISM-TopoMap, установив перед этим симуляционное время в ROS:

`rosparam set use_sim_time true`

`source devel/setup.bash`

`roslaunch prism_topomap build_map_by_iou_scout_rosbag.launch`

4. В третьем терминале запустите rosbag:

`cd /data/bags/30-10-24`

`rosbag play *.bag --clock`

5. В четвертом терминале откройте RViz, чтобы смотреть как строится карта:

`cd src/PRISM-TopoMap/rviz`

`rviz -d test_toposlam_scout_rosbag.rviz`
