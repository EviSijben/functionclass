add_library_unity(gpgomea_fitness OBJECT DiversifiedFCFitness.cpp FCFitness.cpp AccuracyFitness.cpp Fitness.cpp PythonFitness.cpp SymbolicRegressionFitness.cpp SymbolicRegressionLinearScalingFitness.cpp MOFitness.cpp SolutionSizeFitness.cpp DiversifiedSymbolicRegressionFitness.cpp InterpretabilityPHIFitness.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:gpgomea_fitness>
    PARENT_SCOPE)

