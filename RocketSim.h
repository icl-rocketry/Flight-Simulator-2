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
    double drag; // (non-dimensional)

public:
    // constructor
    Rocket(double initialPosition, double initialVelocity, double initialMass);

    // getter functions
    double getPosition() const;
    double getVelocity() const;
    double getAcceleration() const;

    // state update
    void update(double deltaTime);

    // set thrust
    void setThrust(double thrustValue);
};

#endif