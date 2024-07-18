#pragma once

#include <string>
#include <vector>



void parseArguments(int argc, char** argv, std::vector<std::string>& steps);

void printHelp();

void checkPreviousStep(const std::string& current_step);

void executeStep(const std::string& step);