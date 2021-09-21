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


struct tuple{
    int queryIndex;
    int maxIndex;
} tuple;

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
    int* predictions = (int*)malloc(test->num_instances() * sizeof(int));

    // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
    float* candidates = (float*) calloc(k*2, sizeof(float));
    for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

    int num_classes = train->num_classes();

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

        predictions[queryIndex] = max_index;

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

    int indexes[2];

    if(argc != 4)
    {
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
        struct timespec start, end;
        int* predictions = NULL;

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);

        predictions = KNN(train, test, k, 0, test->num_instances());

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);

    }else{
        // Figure out the sections on data by which to operate
        size -= 1;  // do no calculation in main proc
        int numInstances = test->num_instances();
        int section = numInstances / size;
        if(section<=0){
            section = 1;
            size = numInstances;
        }
        struct timespec start, end;

        if(rank==0){
            int* sectionedArray = new int[size+1];
            for(int i = 0; i <= size; i++){sectionedArray[i]=i*section;}
            if(numInstances % size != 0){
                sectionedArray[size] = sectionedArray[size] + (numInstances % size);
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &start);
            // Now that we have all the sections found out, we just need to send the beginning and ending
            // index for the KNN Calculation.
            for(int i = 0; i<size; i++){
                MPI_Send(sectionedArray+i, 2, MPI_INT, i+1, 1,  MPI_COMM_WORLD);
            }
        }else{
            MPI_Recv(&indexes, 2, MPI_INT,0, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            int predLen = indexes[1] - indexes[0];
            int* predictions = (int*)calloc(test->num_instances(), sizeof(int));
            predictions = KNN(train, test, k, indexes[0], indexes[1]);
            // send the predictions array, we can reconstruct the query indexes using the rank
            MPI_Send(&predictions, predLen, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0){
            int* fullPred = (int*)calloc(test->num_instances(), sizeof(int));
            for(int i=1; i<size+1;i++){
                // For every case except the last one the number of predictions will be the section number
                if(i != size-1){
                    int* partialPred = (int*)calloc(section, sizeof(int));
                    MPI_Recv(partialPred, section, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    printf("Partial Predictions: ");
                    for(int i=0; i<80; i++){
                        printf("%d ", partialPred[i]);
                    }
                    printf("\n");
                    // Add predictions to full
                    for(int i=rank*section; i<(rank+1)*section; i++){
                        int temp = partialPred[i-(rank*section)];
                        fullPred[i] = temp;
                    }
                }else{
                    int modulo = numInstances % size;
                    int* partialPred = (int*)calloc((section+modulo), sizeof(int));
                    MPI_Recv(partialPred, (section + modulo), MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    printf("Partial Predictions: ");
                    for(int i=0; i<80; i++){
                        printf("%d ", partialPred[i]);
                    }
                    printf("\n");
                    // Add prediction to full including modulo
                    for(int i=rank*section; i< (rank+1)*section+modulo; i++){
                        int temp = partialPred[i-(rank*section)];
                        fullPred[i] = temp;
                    }
                }
            }
            printf("Coalesced Predictions: ");
            for(int i=0; i<80; i++){
                printf("%d ", fullPred[i]);
            }
            printf("\n");
            // Compute the confusion matrix
            int* confusionMatrix = computeConfusionMatrix(fullPred, test);
            // Calculate the accuracy
            float accuracy = computeAccuracy(confusionMatrix, test);

            printf("The %i-NN classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) 999, accuracy);
        }
    }
    MPI_Finalize();
}
