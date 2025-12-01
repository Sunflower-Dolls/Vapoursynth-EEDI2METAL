cmake_minimum_required(VERSION 3.20)
project(eedi2metalDylib LANGUAGES OBJCXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(VAPOURSYNTH REQUIRED vapoursynth)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(eedi2metal SHARED
    eedi2metal.mm
)

target_include_directories(eedi2metal PRIVATE ${VAPOURSYNTH_INCLUDE_DIRS})

option(DUMP_METAL_BUFFERS "Enable dumping of intermediate Metal buffers" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers for detecting possible errors" OFF)
option(ENABLE_PROFILING "Enable profiling of the kernel" OFF)

if(DUMP_METAL_BUFFERS)
    target_compile_definitions(eedi2metal PRIVATE DUMP_METAL_BUFFERS)
endif()

if(ENABLE_SANITIZERS)
    target_compile_options(eedi2metal PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(eedi2metal PRIVATE -fsanitize=address,undefined)
endif()

if(ENABLE_PROFILING)
    target_compile_definitions(eedi2metal PRIVATE DEBUG_PROFILING)
endif()

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
