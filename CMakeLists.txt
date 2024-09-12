cmake_minimum_required(VERSION 2.8.3)

project(mevius)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

add_message_files(
  FILES
  MeviusLog.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

find_package(roslaunch)

foreach(dir launch models)
  install(DIRECTORY ${dir}/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
