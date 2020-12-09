#ifndef BACKPROPAGATIONNETWORK_HPP
#define BACKPROPAGATIONNETWORK_HPP

#include <vector>
#include "Neuron.h"

namespace NitroNet
{
    class BackpropagationNetwork
    {
    public:
	    BackpropagationNetwork(void);
	    BackpropagationNetwork(const std::vector<unsigned> &topology, const std::vector<TransferFunction> &transferFunctions);
	    void feedForward(const std::vector<double> &inputVals);
	    void backProp(const std::vector<double> &targetVals);
	    void getResults(std::vector<double> &resultVals) const;
	    double getRecentAverageError(void) const;

    private:
	    std::vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	    double m_error;
	    double m_recentAverageError;
	    static double m_recentAverageSmoothingFactor; // Number of training samples to average over
    };
}
#endif
