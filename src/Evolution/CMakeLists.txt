add_library_unity(gpgomea_evolution OBJECT MOArchive.cpp EvolutionRun.cpp EvolutionState.cpp GenerationHandler.cpp NSGA2GenerationHandler.cpp PopulationInitializer.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:gpgomea_evolution>
    PARENT_SCOPE)

