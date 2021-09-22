#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <bits/stdc++.h>
#include "mpi.h"

using namespace std;

float distance(ArffInstance* a, ArffInstance* b) {
    float sum = 0;

    for (int i = 0; i < a->size()-1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    }

    return sum;
}

int* KNN(ArffData* train, ArffData* test, int k, int queryIndex, int queryEnd) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.

    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    int predSize = queryEnd - queryIndex;
    int* predictions = (int*)calloc(predSize, sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

    int num_classes = train->num_classes();
    int baseIndex = queryIndex;

    // Stores bincounts of each class over the final set of candidate NN
    int* classCounts = (int*)calloc(num_classes, sizeof(int));


    for(;queryIndex < queryEnd; queryIndex++){
        for(int keyIndex = 0; keyIndex < train->num_instances(); keyIndex++) {

            float dist = distance(test->get_instance(queryIndex), train->get_instance(keyIndex));

            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(dist < candidates[2*c]){
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }

                    // Set key vector as potential k NN
                    candidates[2*c] = dist;
                    candidates[2*c+1] = train->get_instance(keyIndex)->get(train->num_attributes() - 1)->operator float(); // class value

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k;i++){
            classCounts[(int)candidates[2*i+1]] += 1;
        }

        int max = -1;
        int max_index = 0;
        for(int i = 0; i < num_classes;i++){
            if(classCounts[i] > max){
                max = classCounts[i];
                max_index = i;
            }
        }

        predictions[queryIndex-baseIndex] = max_index;

        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }
        memset(classCounts, 0, num_classes * sizeof(int));
    }

    return predictions;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[]){
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct timespec start, end;
    int* predictions;
    int* counts = NULL;
    int* displacement = NULL;
    int* solution = NULL;

    int solutionSize;
    int index[2];
    int* partialSolution;


    if(argc != 4){
        cout << "Usage: ./main datasets/train.arff datasets/test.arff k" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    if(size == 1){

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        predictions = KNN(train, test, k, 0, test->num_instances());

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);
    }else if(rank==0){
        // Get how large each section to process should be
        int calculatedSize = size;
        int section = test->num_instances() / calculatedSize;
        // If the number of processes is greater than the number of instances, treat the number of sections as 1
        if(section <= 0){
            calculatedSize = test->num_instances();
            section = 1;
        }
        // Create an array representing the boundaries i.e. 2 procs = [0 40 80]
        int* boundaries = new int[calculatedSize+1];
        // 0 is always the starting boundary
        boundaries[0] = 0;
        for(int i=1; i <= calculatedSize; i++){
            boundaries[i] = section*i;
        }
        // If the amount of instances doesn't divide evenly into each process, add the remainder to the final section
        // If there were 3 procs, the array would be [0 26 52 80]
        int modulo = test->num_instances() % calculatedSize;
        if(modulo != 0){
            boundaries[calculatedSize] += modulo;
        }
        // Transmit a set of indexes to each of the helper processes.  [0 26] to 0, [26 52] to 1, [52 80] to 2
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        for(int i=0; i<size; i++){
            MPI_Send(boundaries+i, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }
    // Each process recieves their boundaries to index
    MPI_Recv(&index, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Each process performs KNN on their boundaries and saves the partial KNN Solution
    solutionSize = index[1] - index[0];
    partialSolution = KNN(train, test, k, index[0], index[1]);
    //Gather all sizes of each array
    if(rank==0)
        counts = (int*) malloc(size*sizeof(int));
    MPI_Gather(&solutionSize, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // inside of 0, all KNN are combined in rank order

    if(rank == 0) {
        // create displacement array, each should follow right after the other
        displacement = (int *) malloc(size * sizeof(int));
        displacement[0] = 0;
        int sum = 0;
        for (int i = 1; i < size; i++){
            displacement[i] = counts[i] + sum;
            sum += counts[i];
        }
        // allocate an array for partial solutions
        solution = (int *) malloc(test->num_instances() * sizeof(int));
    }
    // Wait for all KNN to finish before gathering
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(partialSolution, solutionSize, MPI_INT, solution, counts, displacement, MPI_INT,
                0, MPI_COMM_WORLD);
    if(rank == 0){
        // Accuracy is calculated and finals are produced
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(solution, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);

        free(solution);
        free(displacement);
        free(counts);
    }
    MPI_Finalize();
}
