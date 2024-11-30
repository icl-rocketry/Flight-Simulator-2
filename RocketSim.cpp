#include "RocketSim.h"
#include <cmath>

// constructor
Rocket::Rocket(double initialMass, double T, double Cd, double Tb, double Isp)
    : position(0.0), velocity(0.0), acceleration(0.0), mass(initialMass), thrust(T), dragcoeff(Cd), burnTime(Tb), specificImpulse(Isp) {}

// getter functions
double Rocket::getPosition() const { return position; }
double Rocket::getVelocity() const { return velocity; }
double Rocket::getAcceleration() const { return acceleration; }
double Rocket::getMass() const { return mass; }

// state update function
void Rocket::update(double deltaTime) {
    // thrust profile
    if (burnTime > 0.0) {
        thrust = 4000.0;
        burnTime -= deltaTime;
    } else {
        thrust = 0.0;
    }

    // very basic physics model
    double drag = -0.5 * 1.225 * 0.03 * dragcoeff * velocity * abs(velocity);
    double netForce = thrust + drag - 9.81 * mass;
    acceleration = netForce / mass;

    // explicit euler integration
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;

    // update mass
    mass -= thrust / (specificImpulse * 9.81) * deltaTime;
}