#include "TransferFunctions.h"
#include <cmath>

double NitroNet::TransferFunctions::Evaluate(TransferFunction tFunc, double input)
{	
	switch (tFunc)
	{
		case TransferFunction::SIGMOID:
			return sigmoid(input);

		case TransferFunction::LINEAR:
			return linear(input);

		case TransferFunction::GAUSSIAN:
			return gaussian(input);

		case TransferFunction::RATIONALSIGMOID:
			return rationalsigmoid(input);
			
		case TransferFunction::TANH:
			return tanH(input);

		case TransferFunction::NONE:
		default:
			return 0.0;
	}
}
double NitroNet::TransferFunctions::EvaluateDerivative(TransferFunction tFunc, double input)
{
	switch (tFunc)
	{
		case TransferFunction::SIGMOID:
			return sigmoid_derivative(input);

		case TransferFunction::LINEAR:
			return linear_derivative(input);

		case TransferFunction::GAUSSIAN:
			return gaussian_derivative(input);

		case TransferFunction::RATIONALSIGMOID:
			return rationalsigmoid_derivative(input);
			
		case TransferFunction::TANH:
			return tanH_derivate(input);			

		case TransferFunction::NONE:
		default:
			return 0.0;
	}
}

/* Transfer function definitions */
double NitroNet::TransferFunctions::sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NitroNet::TransferFunctions::sigmoid_derivative(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

double NitroNet::TransferFunctions::linear(double x)
{
	return x;
}

double NitroNet::TransferFunctions::linear_derivative(double x)
{
	return 1.0;
}

double NitroNet::TransferFunctions::gaussian(double x)
{
	return exp(-pow(x, 2));
}

double NitroNet::TransferFunctions::gaussian_derivative(double x)
{
	return -2.0 * x * gaussian(x);
}

double NitroNet::TransferFunctions::rationalsigmoid(double x)
{
	return x / (1.0 + sqrt(1.0 + x * x));
}

double NitroNet::TransferFunctions::rationalsigmoid_derivative(double x)
{
	double val = sqrt(1.0 + x * x);
	return 1.0 / (val * (1 + val));
}

double NitroNet::TransferFunctions::tanH(double x)
{
	return tanh(x);
}

double NitroNet::TransferFunctions::tanH_derivate(double x)
{
	return 1.0 - x * x;	
}
