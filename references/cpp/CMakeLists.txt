
cmake_minimum_required(VERSION 3.5)
project(3_fluid)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
set(CMAKE_CXX_FLAGS "-Wall")

# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

# Add default project files
file(GLOB LIBFILES ${PROJECT_SOURCE_DIR}/../include/*.*)
source_group("Library Files" FILES ${LIBFILES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Add your project files
file(GLOB SRCFILES_TEMPLATE *.cpp helper/*.cpp template/*.cpp)
file(GLOB HFILES_TEMPLATE *.h helper/*.h template/*.h)
file(GLOB SRCFILES_SOLUTION *.cpp helper/*.cpp solution/*.cpp)
file(GLOB HFILES_SOLUTION *.h helper/*.h solution/*.h)
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/TestCases.dat" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_definitions(-DIGL_VIEWER_VIEWER_QUIET)
add_executable(3_fluid ${SRCFILES_TEMPLATE} ${HFILES_TEMPLATE} ${LIBFILES} )
target_link_libraries(3_fluid igl::core glfw glad imgui polyscope)
add_executable(3_fluid_solution ${SRCFILES_SOLUTION} ${HFILES_SOLUTION} ${LIBFILES} )
target_link_libraries(3_fluid_solution igl::core glfw glad imgui polyscope)