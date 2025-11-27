cmake_minimum_required(VERSION 3.20)
project(eedi2metalDylib LANGUAGES OBJCXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(eedi2metal SHARED
    eedi2metal.mm
)

target_include_directories(eedi2metal PRIVATE ${VAPOURSYNTH_INCLUDE_DIRECTORY})

set_target_properties(eedi2metal PROPERTIES 
    CXX_FLAGS "-Wno-c23-extensions -mmacosx-version-min=12.0"
    COMPILE_FLAGS "-fobjc-arc"
)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

target_link_libraries(eedi2metal PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)
