// this file builds on the STL vector class to define vector operations
// it is not a class in itself as we are only defining operator overloads - it does not need to store any data

#include "vector.h"
#include <vector>
#include <algorithm>
#include <stdexcept>

// define vector by scalar multiplication
std::vector<double> operator*(const double scalar, std::vector<double> v)
{
    // this is a classic case to use std::transform
    std::vector<double> result(v.size()); // the output will be the same size as the input vectors
    // we need to check if the vector is empty
    if (v.size() == 0)
    {
        throw std::logic_error("Cannot multiply an empty vector by a scalar");
    }
    else
    {
        std::transform(v.begin(), v.end(), result.begin(), [scalar](double &c)
                       { return scalar * c; });
        return result;
    }
}

// define vector addition
std::vector<double> operator+(std::vector<double> v1, const std::vector<double> v2)
{
    // again we can use std::transform using the plus operator from STL
    std::vector<double> result(v1.size()); // the output will be the same size as the input vectors
    // we must check that the vectors are the same size
    if (v1.size() == v2.size())
    {
        std::transform(v1.begin(), v1.end(), v2.begin(), result.begin(), std::plus<double>());
        return result;
    }
    else
    {
        throw std::logic_error("Vectors must be the same size to add them");
    }
}