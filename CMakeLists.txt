cmake_minimum_required(VERSION 2.8.3)
project(simple_loam)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
FIND_PACKAGE(gflags REQUIRED)
INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
add_definitions(${Ceres_DEFINITIONS})

catkin_package(
)

include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${EIGEN3_INCLUDE_DIR}
        ${CERES_LIBRARY_DIRS}
	${PCL_INCLUDE_DIRS})



add_executable(simple_scanRegistration src/scanRegistration.cpp)
target_link_libraries(simple_scanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

link_directories(${PCL_LIBRARY_DIRS} ${CERES_LIBRARY_DIRS})
add_executable(simple_laserOdometry src/laserOdometry.cpp)
add_dependencies(simple_laserOdometry ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PCL_DEFINITIONS} ${Ceres_DEFINITIONS})
target_link_libraries(simple_laserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Ceres_LIBS})


add_executable(simple_laserMapping src/laserMapping.cpp)
target_link_libraries(simple_laserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(simple_transformMaintenance src/transformMaintenance.cpp)
target_link_libraries(simple_transformMaintenance ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

