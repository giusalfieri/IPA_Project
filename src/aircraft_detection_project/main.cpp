#include <iostream>
#include <vector>
#include <string>
#include "pipeline.h" 

int main(int argc, char** argv)
{
    // Parse the command line arguments into steps
    std::vector<std::string> steps;
    parseArguments(argc, argv, steps);

    // If no steps are provided, print the help message and exit
    if (steps.empty())
    {
        printHelp();
        return 0;
    }

    // Execute each step in sequence
    for (const auto& step : steps)
    {
        try
        {
            std::cout << "Executing step \"" << step << "\"\n";
            // Check if the previous step has been completed
            checkPreviousStep(step);
            // Execute the current step
            executeStep(step);
        }
        catch (const std::exception& e)
        {
            // Print an error message if an exception is thrown during step execution
            std::cerr << "Error executing step \"" << step << "\": " << e.what() << "\n";
            return 1;
        }
    }

    return 0;
}