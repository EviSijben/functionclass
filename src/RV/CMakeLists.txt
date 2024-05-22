add_library_unity(gpgomea_rv OBJECT FOS.cpp RV.cpp Optimization.cpp SO_optimization.cpp Tools.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:gpgomea_rv>
    PARENT_SCOPE)

