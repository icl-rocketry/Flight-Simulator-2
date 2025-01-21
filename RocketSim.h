#ifndef ROCKETSIM_H
#define ROCKETSIM_H
#include <vector>

// rocket class
class Rocket {
private:
    std::vector<double> position; // (m)
    std::vector<double> velocity; // (m/s)
    std::vector<double> acceleration; // (m/s^2)
    double mass; // (kg)
    double thrust; // (N)
    double dragcoeff; // (non-dimensional)
    double burnTime; // (s)
    double specificImpulse; // (s)

public:
    // constructor
    Rocket(double initialMass, double T, double Cd, double Tb, double Isp);

    // getter functions
    std::vector<double> getPosition() const;
    std::vector<double> getVelocity() const;
    std::vector<double> getAcceleration() const;
    double getMass() const;

    // state update
    void update(double deltaTime);
};

#endif