//
// Created by evi on 3/22/22.
//

#ifndef GPGOMEA_RV_H
#define GPGOMEA_RV_H


#include "GPGOMEA/Genotype/Node.h"
#include "GPGOMEA/Genotype/SingleNode.h"
typedef struct FOS {
    int length;
    int **sets;
    int *set_length;
} FOS;

class RV {
public:
    RV(Node *tree, std::vector<Operator *> FCs, arma::mat x, arma::mat y, bool get_solutions, size_t nr_per_set, size_t evals, std::string path,  size_t tree_index, size_t tree_nr, size_t data_index, bool calc_constraints);

    double run();
    void saveSolutions();

private:
    Node *tree;
    arma::mat m = arma::mat(100, 1, arma::fill::none);
    std::vector<Operator *> FCs;
    arma::mat x;
    arma::mat y;
    size_t nr_per_set;
    size_t evals;
    bool get_solutions;
    std::string path;
    size_t tree_index;
    size_t tree_nr;
    size_t data_index;
    bool calc_constraints;

    arma::vec range_vec;

private:
    void setParams();

    void interpretCommandLine(int argc, char **argv);

    void generationalStepAllPopulations();

    void generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest);

    int *mergeSortFitness(double *objectives, double *constraints, int number_of_solutions);

    void mergeSortFitnessWithinBounds(double *objectives, double *constraints, int *sorted, int *tosort, int p, int q);

    void mergeSortFitnessMerge(double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q);

    void parseCommandLine(int argc, char **argv);

    void parseOptions(int argc, char **argv, int *index);

    void parseFOSElementSize(int *index, int argc, char **argv);

    void optionError(char **argv, int index);

    void parseParameters(int argc, char **argv, int *index);

    void printUsage();

    void checkOptions();

    void printVerboseOverview();

    void initialize();

    void initializeMemory();

    void initializeNewPopulation();

    void initializeNewPopulationMemory(int population_index);

    void initializeFOS(int population_index);

    void initializeProblem();

    void initializeDistributionMultipliers(int population_index);

    void initializePopulationAndFitnessValues(int population_index);

    void inheritDistributionMultipliers(FOS *new_FOS, FOS *prev_FOS, double *multipliers);

    FOS *learnLinkageTreeRVGOMEA(int population_index);

    void computeRanksForAllPopulations();

    void computeRanksForOnePopulation(int population_index);

    void writeGenerationalStatisticsForOnePopulation(int population_index);

    void writeGenerationalSolutions(short final);

    void writeGenerationalSolutionsBest(short final);

    short checkTerminationCondition();

    short checkSubgenerationTerminationConditions();

    short checkTimeLimitTerminationCondition();

    short checkNumberOfEvaluationsTerminationCondition();

    short checkVTRTerminationCondition();

    void checkAverageFitnessTerminationCondition();

    short checkBestFitnessTerminationCondition();

    void determineBestSolutionInCurrentPopulations(int *population_of_best, int *index_of_best);

    void checkFitnessVarianceTermination();

    short checkFitnessVarianceTerminationSinglePopulation(int population_index);

    void checkDistributionMultiplierTerminationCondition();

    void makeSelections();

    void makePopulation(int population_index);

    void ezilaitiniCovarianceMatrices(int population_index);

    void makeSelectionsForOnePopulation(int population_index);

    void makeSelectionsForOnePopulationUsingDiversityOnRank0(int population_index);

    void estimateParameters(int population_index);

    void estimateMeanVectorML(int population_index);

    void estimateFullCovarianceMatrixML(int population_index);

    void estimateParametersML(int population_index);

    void estimateCovarianceMatricesML(int population_index);

    void initializeCovarianceMatrices(int population_index);

    void copyBestSolutionsToAllPopulations();

    void copyBestSolutionsToPopulation(int population_index);

    void getBestInPopulation(int population_index, int *individual_index);

    void getOverallBest(int *population_index, int *individual_index);

    void evaluateCompletePopulation(int population_index);

    void applyDistributionMultipliersToAllPopulations();

    void applyDistributionMultipliers(int population_index);

    void generateAndEvaluateNewSolutionsToFillAllPopulations();

    void generateAndEvaluateNewSolutionsToFillPopulation(int population_index);

    void computeParametersForSampling(int population_index);

    short generateNewSolutionFromFOSElement(int population_index, int FOS_index, int individual_index, short apply_AMS);

    short applyAMS(int population_index, int individual_index);

    void applyForcedImprovements(int population_index, int individual_index, int donor_index);

    double *generateNewPartialSolutionFromFOSElement(int population_index, int FOS_index);

    short adaptDistributionMultipliers(int population_index, int FOS_index);

    void runAllPopulations();

    short
    generationalImprovementForOnePopulationForFOSElement(int population_index, int FOS_index, double *st_dev_ratio);

    double getStDevRatioForFOSElement(int population_index, double *parameters, int FOS_index);

    void ezilaitini();

    void ezilaitiniMemory();

    void ezilaitiniDistributionMultipliers(int population_index);

    void ezilaitiniParametersForSampling(int population_index);

    void ezilaitiniParametersAllPopulations();


private:
    short print_verbose_overview =0,                              /* Whether to print a overview of settings (0 = no). */
    use_guidelines =0,                                      /* Whether to override parameters with guidelines (for those that exist). */
    fix_seed =1;                                            /* Whether a fixed seed is used. */
    int base_population_size =0,                                /* The size of the first population in the multi-start scheme. */
    *selection_sizes,                                     /* The size of the selection for each population. */
    total_number_of_writes =0,                              /* Total number of times a statistics file has been written. */
    maximum_number_of_populations =0,                       /* The maximum number of populations in the multi-start scheme. */
    number_of_subgenerations_per_population_factor =8,      /* The subgeneration factor in the multi-start scheme. */
    **samples_drawn_from_normal,                           /* The number of samples drawn from the i-th normal in the last generation. */
    **out_of_bounds_draws,                                 /* The number of draws that resulted in an out-of-bounds sample. */
    *no_improvement_stretch,                              /* The number of subsequent generations without an improvement while the distribution multiplier is <= 1.0, for each population separately. */
    maximum_no_improvement_stretch = 0,                      /* The maximum number of subsequent generations without an improvement while the distribution multiplier is <= 1.0. */
    **individual_NIS,
            maximum_number_of_evaluations =0;                                     /* The number of generations a solution has not improved. */
    double                        /* The maximum number of evaluations. */
    maximum_number_of_seconds =0.0,                           /* The maximum number of seconds. */
    tau =0.35,                                                 /* The selection truncation percentile (in [1/population_size,1]). */
    ***populations,                                         /* The populations containing the solutions. */
    **objective_values,                                    /* Objective values for population members. */
    **constraint_values,                                   /* Sum of all constraint violations for population members. */
    **ranks,                                               /* Ranks of population members. */
    ***selections,                                          /* Selected solutions, one for each population. */
    **objective_values_selections,                         /* Objective values of selected solutions. */
    **constraint_values_selections,                        /* Sum of all constraint violations of selected solutions. */
    **distribution_multipliers,                            /* Distribution multipliers of each FOS element of each population. */
    distribution_multiplier_increase =1/0.9,                    /* The multiplicative distribution multiplier increase. */
    distribution_multiplier_decrease =0.9,                    /* The multiplicative distribution multiplier decrease. */
    st_dev_ratio_threshold =1.0,                              /* The maximum ratio of the distance of the average improvement to the mean compared to the distance of one standard deviation before triggering AVS (SDR mechanism). */
    fitness_variance_tolerance =0.0,                          /* The minimum fitness variance level that is allowed. */
    **mean_vectors,                                        /* The mean vectors, one for each population. */
    **mean_shift_vector,                                   /* The mean vectors of the previous generation, one for each population. */
    ****decomposed_covariance_matrices,                      /* The covariance matrices to be used for the sampling. */
    ****decomposed_cholesky_factors_lower_triangle,          /* The unique lower triangular matrix of the Cholesky factorization for every linkage tree element. */
    ***full_covariance_matrix,
            eta_ams = 1.0,
            eta_cov = 1.0;
    FOS **linkage_model;
//    char inputfilePath[200];
//    int resultsPath;

    double elitist_objective_value = 1e308,
            elitist_constraint_value = 1e308;
    std::vector<double> elitist_objective_values;

private:
    // SO_optimization stuff
    short betterFitness(double objective_value_x, double constraint_value_x, double objective_value_y,
                        double constraint_value_y);

    char *installedProblemName(int index);

    int numberOfInstalledProblems();

    void printAllInstalledProblems();

    double installedProblemLowerRangeBound(int index, int dimension);

    double installedProblemUpperRangeBound(int index, int dimension);

    void initializeParameterRangeBounds();

    short isParameterInRangeBounds(double parameter, int dimension);

    void installedProblemEvaluation(int index, double *parameters, double *objective_value, double *constraint_value,
                                    int number_of_touched_parameters, int *touched_parameters_indices,
                                    double *parameters_before, double objective_value_before,
                                    double constraint_value_before);

    void installedProblemEvaluationWithoutRotation(int index, double *parameters, double *objective_value,
                                                   double *constraint_value, int number_of_touched_parameters,
                                                   int *touched_parameters_indices, double *touched_parameters,
                                                   double *parameters_before, double objective_value_before,
                                                   double constraint_value_before);

    void sphereFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    void sphereFunctionPartialProblemEvaluation(double *parameters, double *objective_value, double *constraint_value,
                                                int number_of_touched_parameters, int *touched_parameters_indices,
                                                double *touched_parameters, double *parameters_before,
                                                double objective_value_before, double constraint_value_before);

    double sphereFunctionProblemLowerRangeBound(int dimension);

    double sphereFunctionProblemUpperRangeBound(int dimension);

    void ellipsoidFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    void
    ellipsoidFunctionPartialProblemEvaluation(double *parameters, double *objective_value, double *constraint_value,
                                              int number_of_touched_parameters, int *touched_parameters_indices,
                                              double *touched_parameters, double *parameters_before,
                                              double objective_value_before, double constraint_value_before);

    double ellipsoidFunctionLowerRangeBound(int dimension);

    double ellipsoidFunctionUpperRangeBound(int dimension);

    void cigarFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double cigarFunctionLowerRangeBound(int dimension);

    double cigarFunctionUpperRangeBound(int dimension);

    void tabletFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double tabletFunctionLowerRangeBound(int dimension);

    double tabletFunctionUpperRangeBound(int dimension);

    void cigarTabletFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double cigarTabletFunctionLowerRangeBound(int dimension);

    double cigarTabletFunctionUpperRangeBound(int dimension);

    void twoAxesFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double twoAxesFunctionLowerRangeBound(int dimension);

    double twoAxesFunctionUpperRangeBound(int dimension);

    void
    differentPowersFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double differentPowersFunctionLowerRangeBound(int dimension);

    double differentPowersFunctionUpperRangeBound(int dimension);

    void rosenbrockFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    void
    rosenbrockFunctionPartialProblemEvaluation(double *parameters, double *objective_value, double *constraint_value,
                                               int number_of_touched_parameters, int *touched_parameters_indices,
                                               double *touched_parameters, double *parameters_before,
                                               double objective_value_before, double constraint_value_before);

    double rosenbrockFunctionLowerRangeBound(int dimension);

    double rosenbrockFunctionUpperRangeBound(int dimension);

    void parabolicRidgeFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double parabolicRidgeFunctionLowerRangeBound(int dimension);

    double parabolicRidgeFunctionUpperRangeBound(int dimension);

    void sharpRidgeFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double sharpRidgeFunctionLowerRangeBound(int dimension);

    double sharpRidgeFunctionUpperRangeBound(int dimension);

    void griewankFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double griewankFunctionLowerRangeBound(int dimension);

    double griewankFunctionUpperRangeBound(int dimension);

    void michalewiczFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    void
    michalewiczFunctionPartialProblemEvaluation(double *parameters, double *objective_value, double *constraint_value,
                                                int number_of_touched_parameters, int *touched_parameters_indices,
                                                double *touched_parameters, double *parameters_before,
                                                double objective_value_before, double constraint_value_before);

    double michalewiczFunctionLowerRangeBound(int dimension);

    double michalewiczFunctionUpperRangeBound(int dimension);

    void rastriginFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    void
    rastriginFunctionPartialProblemEvaluation(double *parameters, double *objective_value, double *constraint_value,
                                              int number_of_touched_parameters, int *touched_parameters_indices,
                                              double *touched_parameters, double *parameters_before,
                                              double objective_value_before, double constraint_value_before);

    double rastriginFunctionLowerRangeBound(int dimension);

    double rastriginFunctionUpperRangeBound(int dimension);

    void
    sumOfEllipsoidsFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    void sumOfEllipsoidsFunctionPartialProblemEvaluation(double *parameters, double *objective_value,
                                                         double *constraint_value, int number_of_touched_parameters,
                                                         int *touched_parameters_indices, double *touched_parameters,
                                                         double *parameters_before, double objective_value_before,
                                                         double constraint_value_before);

    double sumOfEllipsoidsFunctionLowerRangeBound(int dimension);

    double sumOfEllipsoidsFunctionUpperRangeBound(int dimension);

    void ciasBRFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double ciasBRFunctionLowerRangeBound(int dimension);

    double ciasBRFunctionUpperRangeBound(int dimension);

    void trapSphereFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double trapSphereFunctionLowerRangeBound(int dimension);

    double trapSphereFunctionUpperRangeBound(int dimension);

    void GrowthLFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthLFunctionLowerRangeBound(int dimension);

    double GrowthLFunctionUpperRangeBound(int dimension);

    void GrowthEFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthEFunctionLowerRangeBound(int dimension);

    double GrowthEFunctionUpperRangeBound(int dimension);

    void GrowthMFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthMFunctionLowerRangeBound(int dimension);

    double GrowthMFunctionUpperRangeBound(int dimension);

    void GrowthGFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthGFunctionLowerRangeBound(int dimension);

    double GrowthGFunctionUpperRangeBound(int dimension);

    void GrowthLOFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthLOFunctionLowerRangeBound(int dimension);

    double GrowthLOFunctionUpperRangeBound(int dimension);

    void GrowthSFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthSFunctionLowerRangeBound(int dimension);

    double GrowthSFunctionUpperRangeBound(int dimension);

    void GrowthBFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GrowthBFunctionLowerRangeBound(int dimension);

    double GrowthBFunctionUpperRangeBound(int dimension);

    void GPExpressionFunctionProblemEvaluation(double *parameters, double *objective_value, double *constraint_value);

    double GPExpressionFunctionLowerRangeBound(int dimension);

    double GPExpressionFunctionUpperRangeBound(int dimension);

private:
    // FOS stuff
    int *mpm_number_of_indices,
            FOS_element_ub,                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
    use_univariate_FOS = 0,                   /* Whether a univariate FOS is used. */
    learn_linkage_tree = 0 ,                   /* Whether the FOS is learned at the start of each generation. */
    static_linkage_tree = 0,                  /* Whether the FOS is fixed throughout optimization. */
    random_linkage_tree =0 ,                  /* Whether the fixed linkage tree is learned based on a random distance measure. */
    FOS_element_size =0;                     /* If positive, the size of blocks of consecutive variables in the FOS. If negative, determines specific kind of linkage tree FOS. */
    double ***MI_matrices,
            **S_matrix,
            *S_vector;                             /* Avoids quadratic memory requirements when a linkage tree is learned based on a random distance measure. */

    void printFOS(FOS *fos);

    FOS *readFOSFromFile(FILE *file);

    FOS *copyFOS(FOS *f);

    FOS *learnLinkageTree(double **covariance_matrix);

    int determineNearestNeighbour(int index, double **S_matrix, int *mpm_number_of_indices, int mpm_length);

    double getSimilarity(int a, int b);

    double **computeMIMatrix(double **covariance_matrix, int n);

    int *matchFOSElements(FOS *new_FOS, FOS *prev_FOS);

    int *hungarianAlgorithm(int **similarity_matrix, int dim);

    void hungarianAlgorithmAddToTree(int x, int prevx, short *S, int *prev, int *slack, int *slackx, int *lx, int *ly,
                                     int **similarity_matrix, int dim);


    void ezilaitiniFOS(FOS *lm);

private:
    // Optimization stuff
    short  black_box_evaluations =0 ,                         /* Whether full (black-box) evaluations must always be performed. */
    use_vtr =0 ,                                       /* Whether to terminate at the value-to-reach (VTR) (0 = no). */
    vtr_hit_status= 0,                                /* Whether the VTR has been reached. */
    *populations_terminated,                        /* Which populations have been terminated. */
    evaluations_for_statistics_hit =0 ,                /* Can be used to write statistics after a certain number of evaluations. */
    write_generational_statistics =0 ,                 /* Whether to compute and write statistics every generation (0 = no). */
    write_generational_solutions = 0;                  /* Whether to write the population every generation (0 = no). */
    int    problem_index =0,                                 /* The index of the optimization problem. */
    number_of_parameters =0,                          /* The number of parameters to be optimized. */
    number_of_populations =0,                         /* The number of parallel populations that initially partition the search space. */
    block_size =0 ,                                    /* The number of variables in one block of the 'sum of rotated ellipsoid blocks' function. */
    number_of_blocks = 0,                              /* The number of blocks the 'sum of rotated ellipsoid blocks' function. */
    block_start = 0,                                   /* The index at which the first block starts of the 'sum of rotated ellipsoid blocks' function. */
    *number_of_generations,                         /* The current generation count of a subgeneration in the interleaved multi-start scheme. */
    total_number_of_generations =0,                   /* The overarching generation count in the interleaved multi-start scheme. */
    *population_sizes;                              /* The size of the population. */
    double //number_of_evaluations,                         /* The current number of times a function evaluation was performed. */
    vtr =0,                                           /* The value-to-reach (function value of best solution that is feasible). */
    rotation_angle =0.0,                                /* The angle of rotation to be applied to the problem. */
    **rotation_matrix,                               /* The rotation matrix to be applied before evaluating. */
    *lower_range_bounds,                            /* The respected lower bounds on parameters. */
    *upper_range_bounds,                            /* The respected upper bounds on parameters. */
    *lower_init_ranges,                             /* The initialization range lower bound. */
    *upper_init_ranges,                             /* The initialization range upper bound */
    lower_user_range=0,                              /* The initial lower range-bound indicated by the user (same for all dimensions). */
    upper_user_range =0;                              /* The initial upper range-bound indicated by the user (same for all dimensions). */

    int number_of_evaluations = 0;

    void initializeObjectiveRotationMatrix( void );
    void ezilaitiniObjectiveRotationMatrix( void );
    double *rotateAllParameters( double *parameters );
    double *rotateParametersInRange( double *parameters, int from, int to );


    void *Malloc( long size );
    double **matrixNew( int n, int m );
    double vectorDotProduct( double *vector0, double *vector1, int n0 );
    double vectorNorm( double *vector0, int n0 );
    double *matrixVectorMultiplication( double **matrix, double *vector, int n0, int n1 );
    double **matrixMatrixMultiplication( double **matrix0, double **matrix1, int n0, int n1, int n2 );
    int blasDSWAP( int n, double *dx, int incx, double *dy, int incy );
    int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy);
    void blasDSCAL( int n, double sa, double x[], int incx );
    int linpackDCHDC( double a[], int lda, int p, double work[], int ipvt[] );
    double **choleskyDecomposition( double **matrix, int n );
    int linpackDTRDI( double t[], int ldt, int n );
    double **matrixLowerTriangularInverse( double **matrix, int n );
    void eigenDecomposition( double **matrix, int n, double **D, double **Q );
    void eigenDecompositionQLalgo2( int n, double **V, double *d, double *e );
    double myhypot( double a, double b );
    void eigenDecompositionHouseholder2( int n, double **V, double *d, double *e );
    void matrixWriteToFile( FILE *file, double **matrix, int n0, int n1 );

    int *mergeSort( double *array, int array_size );
    void mergeSortWithinBounds( double *array, int *sorted, int *tosort, int p, int q );
    void mergeSortWithinBoundsInt( int *array, int *sorted, int *tosort, int p, int q );

    void mergeSortMerge( double *array, int *sorted, int *tosort, int p, int r, int q );
    int *mergeSortInt( int *array, int array_size );
    void mergeSortMergeInt( int *array, int *sorted, int *tosort, int p, int r, int q );

    int *getRanks(double *array, int array_size );
    int *getRanksFromSorted(int *sorted, int array_size );

    double randomRealUniform01( void );
    int randomInt( int maximum );
    double random1DNormalUnit( void );
    double random1DNormalParameterized( double mean, double variance );
    void initializeRandomNumberGenerator( void );
    int *randomPermutation( int n );
    int **allPermutations( int length, int *numberOfPermutations );
    int **allPermutationsSubroutine( int from, int length, int *numberOfPermutations );

    long getMilliSecondsRunning();
    long getMilliSecondsRunningAfterInit();
    long getMilliSecondsRunningSinceTimeStamp( long timestamp );
    long getCurrentTimeStampInMilliSeconds();

    void startTimer( void );
    double getTimer( void );
    void printTimer( void );

    double max( double x, double y );
    double min( double x, double y );
    double distanceEuclidean( double *solution_a, double *solution_b, int n );
    double distanceEuclidean2D( double x1, double y1, double x2, double y2 );

    double *matrixVectorPartialMultiplication( double **matrix, double *vector, int n0, int n1, int number_of_elements, int *element_indices );

    int64_t    random_seed= 611270,                      /* The seed used for the random-number generator. */
    random_seed_changing= 611270;             /* Internally used variable for randomly setting a random seed. */

    long  timestamp_start = 0.0,                       /* The time stamp in milliseconds for when the program was started. */
    timestamp_start_after_init = 0.0;            /* The time stamp in milliseconds for when the algorithm was started */

    double haveNextNextGaussian = 0.0,             /* Internally used variable for sampling the normal distribution. */
    nextNextGaussian = 0.0;                     /* Internally used variable for sampling the normal distribution. */

    #define PI 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798
};

#endif //GPGOMEA_RV_H
