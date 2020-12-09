#include "Neuron.h"
#include <cmath>
#include <random>
#include "Gaussian.h"
#include "TransferFunctions.h"

double NitroNet::Neuron::eta = 0.15; // overall net learning rate
double NitroNet::Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]

NitroNet::Neuron::Neuron(unsigned numOutputs, unsigned myIndex, TransferFunction t)
{
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
	m_transferFunction = t;
}

void NitroNet::Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the nuerons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = 
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double NitroNet::Neuron::sumDOW(const Layer &nextLayer)
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void NitroNet::Neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}
void NitroNet::Neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double NitroNet::Neuron::transferFunction(double x)
{
	// tanh - output range [-1.0..1.0]
	return TransferFunctions::Evaluate(m_transferFunction, x);
	//return tanh(x);
}

double NitroNet::Neuron::transferFunctionDerivative(double x)
{
	// tanh derivative
	return TransferFunctions::EvaluateDerivative(m_transferFunction, x);
	//return 1.0 - x * x;
}

void NitroNet::Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * 
				 prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

void NitroNet::Neuron::setOutputVal(double val)
{
	m_outputVal = val; 
}

double NitroNet::Neuron::getOutputVal() const
{
	return m_outputVal;
}

double NitroNet::Neuron::randomWeight(void) 
{ 
	return Gaussian::GetRandomGaussian();
	//return rand() / double(RAND_MAX); 
}
