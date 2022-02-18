cmake_minimum_required(VERSION 3.18)
project(collision_detection CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

cmake_policy(SET CMP0104 OLD)
set(OpenGL_GL_PREFERENCE GLVND) # to prevent warning from policy CMP0072

# Set this to the minimal version you want to support
find_package(raylib 3.0 QUIET) # Let CMake search for a raylib-config.cmake
# You could change the QUIET above to REQUIRED and remove this if() clause
# This part downloads raylib and builds it if it's not installed on your system
if (NOT raylib_FOUND) # If there's none, fetch and build raylib
    include(FetchContent)
    FetchContent_Declare(
            raylib
            URL https://github.com/raysan5/raylib/archive/master.tar.gz
    )
    FetchContent_GetProperties(raylib)
    if (NOT raylib_POPULATED) # Have we downloaded raylib yet?
        set(FETCHCONTENT_QUIET NO)
        FetchContent_Populate(raylib)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
        # build raylib
        add_subdirectory(${raylib_SOURCE_DIR} ${raylib_BINARY_DIR} EXCLUDE_FROM_ALL) # 防止raylib的静态链接库被安装进系统的/usr/local目录中，参见 https://stackoverflow.com/questions/64900981/how-do-you-prevent-cmake-from-install-ing-targets-from-within-projects-include
    endif()
endif()

if(WIN32)
    find_package(wingetopt QUIET)
    if (NOT wingetopt_FOUND)
        include(FetchContent)
        FetchContent_Declare(
                wingetopt
                URL https://github.com/alex85k/wingetopt/archive/master.tar.gz
        )
        FetchContent_GetProperties(wingetopt)
        if (NOT wingetopt_POPULATED) # Have we downloaded raylib yet?
            set(FETCHCONTENT_QUIET NO)
            FetchContent_Populate(wingetopt)
            add_subdirectory(${wingetopt_SOURCE_DIR} ${wingetopt_BINARY_DIR})
        endif()
    endif()
endif()

#--------------------------------------------------------------------
# Hide the console window in visual studio projects
#--------------------------------------------------------------------
if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

add_executable(collision_detection main.cu scene.cpp scene.h core.cu core.cuh utils.h render.cpp render.h dataGenerator.cpp dataGenerator.h dbg.h dbg.cu rlights.h)

set_target_properties(collision_detection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

if(WIN32)
    target_include_directories(${PROJECT_NAME} PUBLIC ${wingetopt_SOURCE_DIR}/src)
    target_link_libraries(${PROJECT_NAME} raylib wingetopt)
else()
    target_link_libraries(${PROJECT_NAME} raylib)
endif()

# Checks if OSX and links appropriate frameworks (Only required on MacOS)
if (APPLE)
    target_link_libraries(${PROJECT_NAME} "-framework IOKit")
    target_link_libraries(${PROJECT_NAME} "-framework Cocoa")
    target_link_libraries(${PROJECT_NAME} "-framework OpenGL")
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})