// this file builds on the STL vector class to define vector operations
// it is not a class in itself as we are only defining operator overloads - it does not need to store any data

#ifndef VECTOR_H
#define VECTOR_H

#include <vector>

// define vector by scalar multiplication
std::vector<double> operator*(const double scalar, std::vector<double> v);

// define vector addition
std::vector<double> operator+(std::vector<double> v1, const std::vector<double> v2);

// define vector addition
std::vector<double> operator*(std::vector<double> v1, const std::vector<double> v2);

// vector absolute value
std::vector<double> abs(std::vector<double> v);

// vector norm
double norm(std::vector<double> v);


#endif