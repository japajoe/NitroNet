#ifndef NEURON_HPP
#define NEURON_HPP

#include "Layer.h"
#include "TransferFunction.h"
#include <vector>

namespace NitroNet
{
    struct Connection
    {
	    double weight;
	    double deltaWeight;
    };

    class Neuron
    {
    public:
	    TransferFunction m_transferFunction;
	    Neuron(unsigned numOutputs, unsigned myIndex, TransferFunction t);
	    void setOutputVal(double val);
	    double getOutputVal() const;
	    void feedForward(const Layer &prevLayer);
	    void calcOutputGradients(double targetVals);
	    void calcHiddenGradients(const Layer &nextLayer);
	    void updateInputWeights(Layer &prevLayer);
    private:
	    static double eta; // [0.0...1.0] overall net training rate
	    static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	    double transferFunction(double x);
	    double transferFunctionDerivative(double x);
	    // randomWeight: 0 - 1
	    static double randomWeight(void);
	    double sumDOW(const Layer &nextLayer);
	    double m_outputVal;
	    std::vector<Connection> m_outputWeights;
	    unsigned m_myIndex;
	    double m_gradient;
    };
}
#endif
