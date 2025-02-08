#include "RocketSim.h"
#include "RK4.h"
#include "vector.h"
#include <cmath>
#include <vector>

// constructor
Rocket::Rocket(double initialMass, std::vector<double> T, double Cd, double Tb, double Isp)
    : position(0.0), velocity(0.0), acceleration(0.0), mass(initialMass), orientation({0, 0, 1}), thrust(T), dragcoeff(Cd), burnTime(Tb), specificImpulse(Isp) {}

// getter functions
std::vector<double> Rocket::getPosition() const { return position; }
std::vector<double> Rocket::getVelocity() const { return velocity; }
std::vector<double> Rocket::getAcceleration() const { return acceleration; }
std::vector<double> Rocket::getOrientation() const { return orientation; }
double Rocket::getMass() const { return mass; }

// state update function
void Rocket::update(double deltaTime) {
    // thrust profile
    if (burnTime > 0.0) {
        thrust = 4000.0 * orientation;
        burnTime -= deltaTime;
    } else {
        thrust = {0, 0, 0};
    }

    // very basic physics model
    double rho = 1.225 * exp((-1.0 / 10400.0) * position[2]); // simple density lapse model
    std::vector<double> weight = {0, 0, -9.81 * mass};
    std::vector<double> drag = -0.5 * rho * 0.03 * dragcoeff * velocity * abs(velocity); // absolute value ensures drag opposes velocity
    std::vector<double> netForce = thrust + drag + (-1.0) * weight;
    std::vector<double> acceleration = (1.0 / mass) * netForce;

    // explicit euler integration
    velocity = velocity + deltaTime * acceleration;
    position = position + deltaTime * velocity;

    // update mass
    mass = mass - (1 / (specificImpulse * 9.81)) * deltaTime * norm(thrust);
}