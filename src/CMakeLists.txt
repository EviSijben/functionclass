add_definitions(-DGPGOMEA)

set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} -Wextra -Wno-unused-parameter -Wno-redundant-move"
)

add_subdirectory(Evolution)
add_subdirectory(Fitness)
add_subdirectory(Genotype)
add_subdirectory(GOMEA)
add_subdirectory(Operators)
add_subdirectory(RunHandling)
add_subdirectory(Selection)
add_subdirectory(Semantics)
add_subdirectory(Utils)
add_subdirectory(Variation)
add_subdirectory(RV)

set(GPGOMEA_LINK_LIBS ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenMP_CXX_FLAGS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
else()
  set(ALL_COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


# Create C++ executable
add_executable(main main.cpp ${ALL_OBJECT_FILES})
target_link_libraries(main ${GPGOMEA_LINK_LIBS})
target_include_directories(
  main PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


add_executable(params param_calc.cpp ${ALL_OBJECT_FILES})
target_link_libraries(params ${GPGOMEA_LINK_LIBS})
target_include_directories(
        params PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

