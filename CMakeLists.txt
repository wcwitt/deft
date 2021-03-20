cmake_minimum_required(VERSION 3.14)
project(deft LANGUAGES CXX)

include(config.txt)

add_library(deft
	SHARED
    source/box.cpp
    source/fourier.cpp
)
target_include_directories(deft PUBLIC include)

if(DEFT_FFT_TYPE STREQUAL "POCKETFFT")
    target_compile_definitions(deft PUBLIC DEFT_POCKETFFT)
    target_include_directories(deft PUBLIC external/pocketfft)
elseif(DEFT_FFT_TYPE STREQUAL "FFTW")
    target_include_directories(deft PUBLIC ${FFTW_INCLUDE_DIR})
    target_link_libraries(deft PUBLIC "-L${FFTW_LIBRARY_DIR} -lfftw3 -lm")
elseif(DEFT_FFT_TYPE STREQUAL "MKL")
    target_compile_options(deft PUBLIC -m64)
    target_include_directories(deft PUBLIC ${MKL_INCLUDE_DIR} ${MKL_INCLUDE_DIR}/fftw)
    target_link_libraries(deft PUBLIC "-L${MKL_LIBRARY_DIR} -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl")
else()
    message(FATAL_ERROR "DEFT_FFT_TYPE set improperly.")
endif()

if(DEFT_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(deft PUBLIC DEFT_OPENMP)
        target_link_libraries(deft PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "failed to find openmp.")
    endif()
endif()

add_subdirectory(external/pybind11)
pybind11_add_module(
    pydeft
    source/pydeft.cpp
)
target_link_libraries(pydeft PUBLIC deft)
