#include "Gaussian.h"
#include <random>

std::random_device rdev;
std::mt19937 rgen(rdev());
std::uniform_real_distribution<double> idist(0,1);

double NitroNet::Gaussian::GetRandomGaussian()
{
	return idist(rgen);	
}
