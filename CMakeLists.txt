cmake_minimum_required(VERSION 3.20)

project(msi_app_player_subway_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(subway_bot_cpp main.cpp)
target_link_libraries(subway_bot_cpp PRIVATE ${OpenCV_LIBS} user32 gdi32)
