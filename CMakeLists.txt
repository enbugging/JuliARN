cmake_minimum_required(VERSION 3.16)

project(JuliARN VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

include(ExternalProject)

ExternalProject_Add(CGAL
    URL https://github.com/CGAL/cgal/releases/download/v5.5.1/CGAL-5.5.1-library.zip
    #GIT_REPOSITORY https://github.com/CGAL/cgal.git
    #GIT_TAG v5.5.1
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        #-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}/
    SOURCE_DIR   "${CMAKE_BINARY_DIR}/install/cgal"
    BUILD_ALWAYS "" 
    INSTALL_COMMAND "" 
)

list(APPEND includePath 
    "${CMAKE_BINARY_DIR}/install/cgal/include"
)

if (WIN32)
    ExternalProject_Add(cgal_dependencies
        URL https://github.com/CGAL/cgal/releases/download/v5.5.1/CGAL-5.5.1-win64-auxiliary-libraries-gmp-mpfr.zip
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            #-DCMAKE_INSTALL_PREFIX:PATH="${CMAKE_BINARY_DIR}/install"
        SOURCE_DIR   "${CMAKE_BINARY_DIR}/install/cgal/auxiliary"
        #INSTALL_DIR   "${CMAKE_INSTALL_PREFIX}/install"
        CONFIGURE_COMMAND "" 
        BUILD_COMMAND "" 
        INSTALL_COMMAND "" 
    )

    set(BOOST_BOOTSTRAP_CMD ${CMAKE_CURRENT_BINARY_DIR}/boost-prefix/src/boost/bootstrap.bat)
    set(BOOST_BUILD_CMD ${CMAKE_CURRENT_BINARY_DIR}/boost-prefix/src/boost/b2.exe)

    ExternalProject_Add(boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.zip
        #GIT_REPOSITORY https://github.com/boostorg/boost.git
        #GIT_TAG boost-1.78.0
        CONFIGURE_COMMAND ${BOOST_BOOTSTRAP_CMD}
        BUILD_COMMAND
            ${BOOST_BUILD_CMD}
            --prefix=${CMAKE_INSTALL_PREFIX}
            -sZLIB_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib
            --without-python
            address-model=64
            variant=debug,release
            link=shared
            runtime-link=shared
            threading=multi
            install
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )

    add_dependencies(CGAL boost)
    add_dependencies(CGAL cgal_dependencies)

    list(APPEND includePath
        "${CMAKE_BINARY_DIR}/install/cgal/auxiliary/gmp/include"
        "${CMAKE_BINARY_DIR}/install/boost/include/boost-1_81/"
    )
endif()

add_executable(sandbox ./sandbox/sandbox.cpp)
add_dependencies(sandbox CGAL)
target_include_directories(sandbox PRIVATE "$<BUILD_INTERFACE:${includePath}>")
