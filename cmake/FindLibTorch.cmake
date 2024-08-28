if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(TORCH_BUILD_TYPE "debug")
elseif(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    set(TORCH_BUILD_TYPE "release")
else()
    set(TORCH_BUILD_TYPE "debug")
endif()

set(LIBTORCH_VERSION "2.3.1")
set(LIBTORCH_CUDA "118")

if (WIN32)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-debug-${LIBTORCH_VERSION}%2Bcu${LIBTORCH_CUDA}.zip")
    else()
        set(TORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-${LIBTORCH_VERSION}%2Bcu${LIBTORCH_CUDA}.zip")
    endif()
elseif(LINUX)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip")
elseif(APPLE)
        set(TORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip")
endif()

find_package(Torch QUIET CONFIG)
if(NOT Torch_FOUND)
    message(STATUS "libtorch not found")
    message(STATUS "Fetching libtorch")
    include(FetchContent)
    FetchContent_Declare(
        libtorch
        URL ${TORCH_URL}
        SOURCE_DIR libtorch)
    FetchContent_GetProperties(libtorch)
    if(NOT libtorch_POPULATED)
        unset(FETCHCONTENT_QUIET CACHE)
        FetchContent_Populate(libtorch)
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
        set(LIBTORCH_DIR
            ${CMAKE_BINARY_DIR}/libtorch
        )

        set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")

    endif()
    find_package(Torch REQUIRED)
else()
    message(STATUS "libtorch found")
endif()

# get all .dll files in ${CMAKE_BINARY_DIR}/libtorch/lib
file(GLOB TORCH_DLLS "${CMAKE_BINARY_DIR}/libtorch/lib/*.dll")

# install the dll files to the binary directory for taregt "test"
install(FILES ${TORCH_DLLS} DESTINATION obs-plugins/64bit)