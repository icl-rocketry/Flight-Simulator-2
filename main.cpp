#include <iostream>
#include "RocketSim.h"

int main() {
    // parameters
    double timeStep = 0.02; // timestep (s)
    double maxTime = 100.0; // max time (s)

    // initialise (m, T, CD, Tb)
    Rocket rocket(60.0, 4000.0, 0.5, 6.0, 200);

    // actually run the simulation (and stop if rocket hits the ground)
    for (double t = 0.0; t < maxTime && rocket.getPosition() >= 0; t += timeStep) {
        rocket.update(timeStep);

        // print state
        std::cout << "Time: " << t 
                  << " s, Position: " << rocket.getPosition() 
                  << " m, Velocity: " << rocket.getVelocity() 
                  << " m/s, Acceleration: " << rocket.getAcceleration() 
                  << " m/s^2, Mass: " << rocket.getMass() << std::endl;
    }

    return 0;
}