cmake_minimum_required(VERSION 3.13)

project(openWakeWord C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

ADD_EXECUTABLE(openWakeWord ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wl,-rpath,'$ORIGIN'")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

target_link_libraries(openWakeWord
                      pthread
                      onnxruntime)

# Example: openWakeWord-native/lib/x86_64
target_link_directories(openWakeWord PUBLIC
                        ${CMAKE_CURRENT_LIST_DIR}/../../lib/${CMAKE_HOST_SYSTEM_PROCESSOR})

target_include_directories(openWakeWord PUBLIC
                           ${CMAKE_CURRENT_LIST_DIR}/../include)
