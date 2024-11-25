#ifndef ROCKETSIM_H
#define ROCKETSIM_H

// rocket class
class Rocket {
private:
    double position; // (m)
    double velocity; // (m/s)
    double acceleration; // (m/s^2)
    double mass; // (kg)
    double thrust; // (N)
    double dragcoeff; // (non-dimensional)
    double burnTime; // (s)#
    double specificImpulse; // (s)

public:
    // constructor
    Rocket(double initialMass, double T, double CD, double Tb, double Isp);

    // getter functions
    double getPosition() const;
    double getVelocity() const;
    double getAcceleration() const;
    double getMass() const;

    // state update
    void update(double deltaTime);
};

#endif