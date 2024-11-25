#include "RocketSim.h"

// constructor
Rocket::Rocket(double initialPosition, double initialVelocity, double initialMass)
    : position(initialPosition), velocity(initialVelocity), acceleration(0.0), mass(initialMass), thrust(0.0), drag(0.0) {}

// getter functions
double Rocket::getPosition() const { return position; }
double Rocket::getVelocity() const { return velocity; }
double Rocket::getAcceleration() const { return acceleration; }

// constant thrust for now
void Rocket::setThrust(double thrustValue) { thrust = thrustValue; }

// state update function
void Rocket::update(double deltaTime) {
    // very basic physics model
    double netForce = thrust - drag;
    acceleration = netForce / mass;

    // explicit euler integration
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;
}