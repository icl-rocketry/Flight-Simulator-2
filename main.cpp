#include <iostream>
#include <fstream>
#include <vector>
#include "RocketSim.h"

int main() {
    // parameters
    double timeStep = 0.01; // timestep (s)
    double maxTime = 100.0; // max time (s)

    // initialise (m, T, CD, Tb, Isp)
    Rocket rocket(79.0, 4096.0, 0.5, 10.0, 200);

    // set up output file
    std::ofstream file("flight_data.txt");

    // important values to print to console
    double apogee = 0;
    double maxSpeed = 0;
    double maxAcceleration = 0;

    // actually run the simulation (and stop if rocket hits the ground)
    for (double t = 0.0; t < maxTime && rocket.getPosition() >= 0; t += timeStep) {
        rocket.update(timeStep);

        // get values
        std::vector<double> pos = rocket.getPosition();
        std::vector<double> vel = rocket.getVelocity();
        std::vector<double> acc = rocket.getAcceleration();

        // print state
        file << t 
                  << ", " << pos
                  << ", " << vel
                  << ", " << acc
                  << ", " << rocket.getMass() << std::endl;

        // update apogee and other key flight values
        if (pos > apogee) {
            apogee = pos;
        }
        if (vel > maxSpeed) {
            maxSpeed = vel;
        }
        if (acc > maxAcceleration) {
            maxAcceleration = acc;
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