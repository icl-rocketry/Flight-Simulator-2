#include <iostream>
#include "RocketSim.h"

int main() {
    // parameters
    double timeStep = 0.01; // timestep (s)
    double maxTime = 10.0; // max time (s)

    // initialise (z, v, m)
    Rocket rocket(0.0, 0.0, 100.0);

    // thrust (constant for now)
    rocket.setThrust(500.0);

    // actually run the simulation
    for (double t = 0.0; t < maxTime; t += timeStep) {
        rocket.update(timeStep);

        // print state
        std::cout << "Time: " << t 
                  << " s, Position: " << rocket.getPosition() 
                  << " m, Velocity: " << rocket.getVelocity() 
                  << " m/s, Acceleration: " << rocket.getAcceleration() 
                  << " m/s^2\n";
    }

    return 0;
}