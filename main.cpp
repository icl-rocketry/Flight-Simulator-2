#include <iostream>
#include <fstream>
#include "RocketSim.h"

int main() {
    // parameters
    double timeStep = 0.02; // timestep (s)
    double maxTime = 100.0; // max time (s)

    // initialise (m, T, CD, Tb)
    Rocket rocket(60.0, 4000.0, 0.5, 6.0, 200);

    // set up output file
    std::ofstream file("flight_data.txt");

    // actually run the simulation (and stop if rocket hits the ground)
    for (double t = 0.0; t < maxTime && rocket.getPosition() >= 0; t += timeStep) {
        rocket.update(timeStep);

        // print state
        file << t 
                  << ", " << rocket.getPosition() 
                  << ", " << rocket.getVelocity() 
                  << ", " << rocket.getAcceleration() 
                  << ", " << rocket.getMass() << std::endl;
    }

    file.close();

    return 0;
}