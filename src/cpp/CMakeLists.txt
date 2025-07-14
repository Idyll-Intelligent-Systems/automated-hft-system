cmake_minimum_required(VERSION 3.20)
project(HFTTradingSystem VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fno-plt -fno-semantic-interposition")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=thin")
endif()

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(Boost REQUIRED COMPONENTS system thread chrono)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/network
    ${CMAKE_CURRENT_SOURCE_DIR}/data
    ${CMAKE_CURRENT_SOURCE_DIR}/strategy
    ${CMAKE_CURRENT_SOURCE_DIR}/risk
    ${CMAKE_CURRENT_SOURCE_DIR}/order
    ${CMAKE_CURRENT_SOURCE_DIR}/gateway
)

# Common library
add_subdirectory(common)
add_subdirectory(network)
add_subdirectory(data)
add_subdirectory(strategy)
add_subdirectory(risk)
add_subdirectory(order)
add_subdirectory(gateway)

# Main trading engine executable
add_executable(trading_engine
    main.cpp
)

target_link_libraries(trading_engine
    common
    network
    data
    strategy
    risk
    order
    gateway
    Boost::system
    Boost::thread
    Boost::chrono
    TBB::tbb
    fmt::fmt
    spdlog::spdlog
)

# Optimization flags for production builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(trading_engine PRIVATE PRODUCTION_BUILD)
    set_target_properties(trading_engine PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

# Installation
install(TARGETS trading_engine
    RUNTIME DESTINATION bin
)
