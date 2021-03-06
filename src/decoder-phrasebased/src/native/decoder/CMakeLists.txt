cmake_minimum_required(VERSION 3.2)
project(mmt_pbdecoder)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC")

# moses defines
add_definitions(-DMAX_NUM_FACTORS=1 -DWITH_THREADS -DBOOST_TEST_DYN_LINK -DTRACE_ENABLE=1)
# note: MOSES_VERSION_ID is defined in moses/CMakeLists.txt
# "use TRACE_ENABLE to turn on output of any debugging info" (moses/Util.h)


# Includes
include_directories(.)
include_directories(moses)

# Binaries

add_library(${PROJECT_NAME} SHARED $<TARGET_OBJECTS:moses_moses>
        $<TARGET_OBJECTS:moses_kenlm_util> $<TARGET_OBJECTS:moses_java>
        $<TARGET_OBJECTS:mmt_logging> $<TARGET_OBJECTS:mmt_vocabulary>)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

add_executable(moses-main executables/moses-main.cpp)
target_link_libraries(moses-main ${Boost_LIBRARIES} ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} mmt_ilm mmt_sapt)


# External Libraries

## Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS
        system
        thread
        filesystem
        unit_test_framework
        program_options
        iostreams
        )
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} pthread)

## JNI
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

## ZLIB
# only for util/read_compressed.cc
find_package(ZLIB REQUIRED)
add_definitions(-DHAVE_ZLIB=1)
target_link_libraries(${PROJECT_NAME} ${ZLIB_LIBRARIES})

## RocksDB
find_package(Rocksdb REQUIRED)
include_directories(${Rocksdb_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${Rocksdb_LIBS})

# Sources

# note: include_directories() only affects later subdirectories.

add_subdirectory(moses)
add_subdirectory(java)
add_subdirectory(util)  # KenLM utils, used mostly by MERT
add_subdirectory(mert)
