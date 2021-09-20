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

using namespace std;

pthread_mutex_t predlock;
pthread_mutex_t classlock;

struct knnVals{
    int* predictions;
    int num_classes;
    int* classCounts;
} knnVals;

struct parameters{
    struct knnVals* vals;
    ArffData* train;
    ArffData* test;
    int k;
    int queryIndex;
    int queryEnd;
    int numInstances;
} parameters;

float distance(ArffInstance* a, ArffInstance* b) {
    float sum = 0;

    for (int i = 0; i < a->size()-1; i++) {
        float diff = (a->get(i)->operator float() - b->get(i)->operator float());
        sum += diff*diff;
    }

    return sum;
}
void* innerKNN(void * parameters){
    struct parameters* myParams = (struct parameters*) parameters;
    struct knnVals* val = myParams->vals;
    ArffData* train = myParams->train;
    ArffData* test = myParams->test;
    int k = myParams->k;
    int queryIndex = myParams->queryIndex;
    int queryEnd = myParams->queryEnd;
    int numInstances = myParams->numInstances;

    for(;queryIndex < queryEnd; queryIndex++){
        // stores k-NN candidates for a query vector as a sorted 2d array. First element is inner product, second is class.
        float* candidates = (float*) calloc(k*2, sizeof(float));
        for(int i = 0; i < 2*k; i++){ candidates[i] = FLT_MAX; }

        for(int keyIndex = 0; keyIndex < numInstances; keyIndex++){
            float dist = distance(test->get_instance(queryIndex), train->get_instance(keyIndex));

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
        pthread_mutex_lock(&classlock);
        for(int i = 0; i < k;i++){
            val->classCounts[(int)candidates[2*i+1]] += 1;
        }
        pthread_mutex_unlock(&classlock);

        int max = -1;
        int max_index = 0;

        for(int i = 0; i < val->num_classes;i++){
            if(val->classCounts[i] > max){
                pthread_mutex_lock(&classlock);
                max = val->classCounts[i];
                pthread_mutex_unlock(&classlock);
                max_index = i;
            }
        }

        pthread_mutex_lock(&predlock);
        val->predictions[queryIndex] = max_index;
        pthread_mutex_unlock(&predlock);

        pthread_mutex_lock(&classlock);
        memset(val->classCounts, 0, val->num_classes * sizeof(int));
        pthread_mutex_unlock(&classlock);

        free(candidates);
    }
    return NULL;
}
int* KNN(ArffData* train, ArffData* test, int k, int numThreads) {
    // Implements a sequential kNN where for each candidate query an in-place priority queue is maintained to identify the kNN's.
    struct knnVals val;
    struct knnVals* ptr_val;
    // predictions is the array where you have to return the class predicted (integer) for the test dataset instances
    val.predictions = (int*)malloc(test->num_instances() * sizeof(int));
    val.num_classes = train->num_classes();
    // Stores bincounts of each class over the final set of candidate NN
    val.classCounts = (int*)calloc(val.num_classes, sizeof(int));
    ptr_val = &val;

    // tid
    pthread_t* threads = (pthread_t*)malloc(numThreads*sizeof(pthread_t));

    int numInstances = test->num_instances();
    // Section query based on number of threads
    // If the number of threads is greater than the number of instances, only utilize to the number of instances
    int section =  numInstances / numThreads;
    if(section <= 0){
        section = 1;
        numThreads = numInstances;
    }
    int* sectionedArray = new int[numThreads+1];

    for(int i=0; i <= numThreads; i++){
        sectionedArray[i] = i * section;
    }
    if(numInstances % numThreads == 0){
        sectionedArray[numThreads-1] += numInstances % numThreads;
    }

    // Run each thread on the sections
    for(int i=0; i<numThreads; i++){
        struct parameters p;
        p.vals = ptr_val;
        p.train = train;
        p.test = test;
        p.k = k;
        p.queryIndex = sectionedArray[i];
        p.queryEnd = sectionedArray[i+1];
        p.numInstances = p.test->num_instances();

        struct parameters* pPtr;
        pPtr = &p;
        pthread_create(&threads[i], NULL, innerKNN, (void*) pPtr);
    }
    // join them to main before predictions
    for(int i=0; i < numThreads; i++){
        pthread_join(threads[i], NULL);
    }
    return val.predictions;
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

    if(argc != 5)
    {
        cout << "Usage: ./main datasets/train.arff datasets/test.arff k numThreads" << endl;
        exit(0);
    }

    int k = strtol(argv[3], NULL, 10);
    int numThreads = strtol(argv[4], NULL, 10);
    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    struct timespec start, end;
    int* predictions = NULL;


    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    predictions = KNN(train, test, k, numThreads);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The %i-NN %i-Thread classifier for %lu test instances on %lu train instances required %llu ms CPU time. Accuracy was %.4f\n", k, numThreads,
           test->num_instances(), train->num_instances(), (long long unsigned int) diff, accuracy);
}
