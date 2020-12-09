#ifndef TRANSFERFUNCTIONS_HPP
#define TRANSFERFUNCTIONS_HPP

#include "TransferFunction.h"

namespace NitroNet
{
    class TransferFunctions
    {
    public:
	    static double Evaluate(TransferFunction tFunc, double input);
	    static double EvaluateDerivative(TransferFunction tFunc, double input);
    private:
	    /* Transfer function definitions */
	    static double sigmoid(double x);
	    static double sigmoid_derivative(double x);
	    static double linear(double x);
	    static double linear_derivative(double x);
	    static double gaussian(double x);
	    static double gaussian_derivative(double x);
	    static double rationalsigmoid(double x);
	    static double rationalsigmoid_derivative(double x);
	    static double tanH(double x);
	    static double tanH_derivate(double x);
    };
}
#endif
