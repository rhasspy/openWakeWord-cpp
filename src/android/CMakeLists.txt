# https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android/1.16.3

cmake_minimum_required(VERSION 3.22.1)

project(appmodules C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra") # -Wl,-rpath,'$ORIGIN'
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")


set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)


add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
    onnxruntime
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/../../../onnxruntime-android/jni/${ANDROID_ABI}/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/../../../onnxruntime-android/headers
)


find_library(log-lib log)
find_library(android-lib android)


add_library(
    openWakeWord
    SHARED
    ${CMAKE_CURRENT_LIST_DIR}/../main.cpp
)

target_link_libraries(
    openWakeWord
    PRIVATE
    Threads::Threads
    onnxruntime
    aaudio
    log
    ${log-lib}
    ${android-lib}
)
