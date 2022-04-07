cmake_minimum_required(VERSION 3.0.0)
project(testmvaqcuire VERSION 0.1.0)

include(CTest)
enable_testing()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include($ENV{MVIMPACT_ACQUIRE_DIR}/mvIMPACT_AcquireConfig.cmake)

add_executable(testmvaqcuire main.cpp)
target_link_libraries(testmvaqcuire PRIVATE ${mvIMPACT_Acquire_LIBRARIES} Threads::Threads)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
