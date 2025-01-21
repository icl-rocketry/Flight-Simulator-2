#include "RocketSim.h"
#include "RK4.h"
#include "vector.h"
#include <cmath>
#include <vector>

// constructor
Rocket::Rocket(double initialMass, double T, double Cd, double Tb, double Isp)
    : position(0.0), velocity(0.0), acceleration(0.0), mass(initialMass), thrust(T), dragcoeff(Cd), burnTime(Tb), specificImpulse(Isp) {}

// getter functions
std::vector<double> Rocket::getPosition() const { return position; }
std::vector<double> Rocket::getVelocity() const { return velocity; }
std::vector<double> Rocket::getAcceleration() const { return acceleration; }
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
    double rho = 1.225 * exp(-position / 10400); // simple density lapse model
    std::vector<double> drag = -0.5 * rho * 0.03 * dragcoeff * velocity * abs(velocity); // absolute value ensures drag opposes velocity
    std::vector<double> netForce = thrust + drag - 9.81 * mass;
    std::vector<double> acceleration = netForce / mass;

    // explicit euler integration
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;

    // update mass
    mass -= thrust / (specificImpulse * 9.81) * deltaTime;
}