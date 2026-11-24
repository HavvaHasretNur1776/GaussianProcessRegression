cmake_minimum_required(VERSION 3.16)
project(BPairGenerator LANGUAGES CXX)
if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
find_package(ROOT CONFIG REQUIRED COMPONENTS Core RIO Tree Physics)
find_package(EvtGen CONFIG REQUIRED)
add_executable(BPairGenerator.exe BPairGenerator.cc models/EvtKnunu.cc)
target_compile_features(BPairGenerator.exe PRIVATE cxx_std_17)
target_include_directories(BPairGenerator.exe PRIVATE models)
target_link_libraries(BPairGenerator.exe PRIVATE
  ROOT::Core ROOT::RIO ROOT::Tree ROOT::Physics EvtGen::EvtGen EvtGen::EvtGenExternal)
target_compile_definitions(BPairGenerator.exe PRIVATE
  BPAIR_DEFAULT_DATA_DIR="${EVTGEN_DATA_DIR}"
  BPAIR_DEFAULT_USER_DECAY="${CMAKE_CURRENT_SOURCE_DIR}/signal.dec")
target_compile_options(BPairGenerator.exe PRIVATE -Wall -Wextra -Wpedantic)
