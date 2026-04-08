cmake_minimum_required(VERSION 3.16)
project(cufsm LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# for nanobind
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

if (SKBUILD)
  message(STATUS "Configuring via scikit-build-core. i.e. Installing via pip.")
  find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
    OPTIONAL_COMPONENTS Development.SABIModule
  )

else()
  message(STATUS "Configuring via cmake directly. If this is not what you want, please install using pip:
    =====================================================================================================
      pip install .
    =====================================================================================================")
  
  set(Python_ROOT_DIR $ENV{HOME}/miniconda3/envs/puzzle/)
  find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
  )
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  
endif()
find_package(nanobind CONFIG REQUIRED)

include_directories(
    include
)

set(SRC_FILES
  src/fsm_cpu.cpp
)

set(CUDA_FILES
  src/fsm_gpu.cu
)

# add_executable(${PROJECT_NAME} src/main.cu ${SRC_FILES} ${CUDA_FILES})
# add_executable(${PROJECT_NAME} src/main2.cpp ${SRC_FILES} ${CUDA_FILES})
# set_target_properties(${PROJECT_NAME} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     # CUDA_RESOLVE_DEVICE_SYMBOLS ON
# )
# target_compile_definitions(${PROJECT_NAME} PRIVATE USE_CUDA)
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
# target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX CUDA::cudart)

nanobind_add_module(cufsm_nb src/cufsm_nb.cpp NOMINSIZE ${SRC_FILES} ${CUDA_FILES})
set_target_properties(cufsm_nb PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  # CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(cufsm_nb PRIVATE USE_CUDA)
target_compile_options(cufsm_nb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler -fopenmp>)
target_link_libraries(cufsm_nb PRIVATE OpenMP::OpenMP_CXX CUDA::cudart)
target_include_directories(cufsm_nb PRIVATE include)
install(TARGETS cufsm_nb LIBRARY DESTINATION cufsm)