cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(HornFocusing)

find_package(Geant4 REQUIRED ui_all vis_all)
find_package(Threads REQUIRED)

# Geant4 include directories and compile definitions
include(${Geant4_USE_FILE})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Explicitly list all source files
set(SOURCES
    src/DetectorConstruction.cc
    src/PhysicsList.cc
    src/ActionInitialization.cc
    src/PrimaryGeneratorAction.cc
    src/RunAction.cc
    src/EventAction.cc
    src/SteppingAction.cc
    src/HornField.cc
    src/Cli.cc
)

# Add the executable with explicit source files
add_executable(horn_sim horn_sim.cc ${SOURCES})

# Link against Geant4 libraries
target_link_libraries(horn_sim ${Geant4_LIBRARIES} Threads::Threads)

# Install the executable
install(TARGETS horn_sim DESTINATION bin)

# Copy necessary scripts to build directory
set(HORN_SCRIPTS
    init_vis.mac
    vis.mac
    run.mac
)

foreach(_script ${HORN_SCRIPTS})
  configure_file(
    ${PROJECT_SOURCE_DIR}/${_script}
    ${PROJECT_BINARY_DIR}/${_script}
    COPYONLY
  )
endforeach()
