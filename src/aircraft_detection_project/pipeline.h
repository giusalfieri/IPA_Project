#pragma once

#include <string>
#include <vector>



void extract_csv_for_svm_training();

void extractTemplates();

void performKMeansBySize();

void performKMeansByIntensity();

void resizeImagesAcrossClusters();

void generateEigenplanes();

void classify(const std::string& testing_img_id);

void evaluatePerformance();

void printHelp();

void parseArguments(int argc, char** argv, std::vector<std::string>& steps);

void checkPreviousStep(const std::string& current_step);

void executeStep(const std::string& step);