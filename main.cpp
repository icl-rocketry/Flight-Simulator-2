#include <iostream>
#include <fstream>
#include <vector>
#include "RocketSim.h"
#include "vector.h"

int main() {
    // parameters
    double timeStep = 0.01; // timestep (s)
    double maxTime = 100.0; // max time (s)

    // initialise (m, T, CD, Tb, Isp)
    Rocket rocket(79.0, {0, 0, 4096.0}, 0.5, 10.0, 200);

    // set up output file
    std::ofstream file("flight_data.txt");

    // important values to print to console
    double apogee = 0;
    double maxSpeed = 0;
    double maxAcceleration = 0;

    // actually run the simulation (and stop if rocket hits the ground)
    for (double t = 0.0; t < maxTime && rocket.getPosition()[2] >= 0; t += timeStep) {
        rocket.update(timeStep);

        // get values
        std::vector<double> pos = rocket.getPosition();
        std::vector<double> vel = rocket.getVelocity();
        std::vector<double> acc = rocket.getAcceleration();

        // print state
        file << t 
                  << ", " << pos[0] << pos[1] << pos[2]
                  << ", " << vel[0] << vel[1] << vel[2]
                  << ", " << acc[0] << acc[1] << acc[2]
                  << ", " << rocket.getMass() << std::endl;

        // update apogee and other key flight values
        if (pos[2] > apogee) {
            apogee = pos[2];
        }
        if (norm(vel) > maxSpeed) {
            maxSpeed = norm(vel);
        }
        if (norm(acc) > maxAcceleration) {
            maxAcceleration = norm(acc);
        }
    }

    file.close();

    // display important values
    std::cout.precision(3);
    std::cout << "Apogee: " << apogee << " m" << std::endl;
    std::cout << "Max. Speed: " << maxSpeed << " m/s" << std::endl;
    std::cout << "Max. Acceleration: " << maxAcceleration << " m/s^2" << std::endl;

    return 0;
}