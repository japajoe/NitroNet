#include <iostream>
#include "BackpropagationNetwork.h"
#include "TransferFunction.h"
#include <vector>

using namespace NitroNet;

std::vector<unsigned> layerSizes;	
std::vector<TransferFunction> transferFunctions;	
BackpropagationNetwork network;
std::vector<double> input;	
std::vector<double> desired;
std::vector<double> results;
double error = 1;

int main()
{
    layerSizes.push_back(3);
	layerSizes.push_back(4);
	layerSizes.push_back(4);
	layerSizes.push_back(2);	
	
	transferFunctions.push_back(TransferFunction::NONE);
	transferFunctions.push_back(TransferFunction::TANH);
	transferFunctions.push_back(TransferFunction::TANH);
	transferFunctions.push_back(TransferFunction::TANH);
	
	network = BackpropagationNetwork(layerSizes, transferFunctions);
	
	input.push_back(0);
	input.push_back(1);
	input.push_back(1);
	
	desired.push_back(1);
	desired.push_back(1);

    while(error > 0.001)
	{
		network.feedForward(input);
		network.getResults(results);
		network.backProp(desired);
		error = network.getRecentAverageError();
	}

    std::cout << "Learning error: " << error << std::endl;

	return 0;
}
