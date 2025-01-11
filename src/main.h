#ifndef MAIN_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define MAIN_H

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include <jni.h>

void ensureArg(int argc, char *argv[], int argi);
void printUsage(char *argv[]);
int main(int argc, char *argv[]);

#endif