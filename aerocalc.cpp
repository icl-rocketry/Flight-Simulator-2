#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#define _USE_MATH_DEFINES
#include <cmath>
#include <tuple>
#include <numbers>
#include <algorithm>
#include <string>
#include <sstream>
#include <optional>


class RegularGridInterpolator {
public:
    RegularGridInterpolator(const std::vector<std::vector<double>>& grid_points, // takes 2x2 vector grid coordinates which is the grid it is interpolating
                            const std::vector<double>& grid_values) // takes value of every point in the grid
        : grid_points_(grid_points), grid_values_(grid_values), dim_(grid_points.size()) {
        
        
        size_t product = 1;
        for (const auto& dim : grid_points) {
            product *= dim.size(); // dim is number of dimensions of the data 
        }
        
    }

    double interpolate(const std::vector<double>& point) const { // takes a given point to interpolate 
        // Find indices and interpolation weights for each dimension
        std::vector<int> lower_indices(dim_);
        std::vector<double> weights(dim_);
        for (size_t d = 0; d < dim_; ++d) {
            auto [lower, upper, weight] = find_bounds_and_weights(grid_points_[d], point[d]);
            lower_indices[d] = lower;
            weights[d] = weight;
        }

        // Perform multidimensional linear interpolation
        return multilinear_interpolation(lower_indices, weights, 0, 1);
    }

private:
    const std::vector<std::vector<double>>& grid_points_;
    const std::vector<double>& grid_values_; 
    size_t dim_;

    // Find bounds and interpolation weight for a single dimension
    static std::tuple<int, int, double> find_bounds_and_weights(const std::vector<double>& grid,
                                                                double value) {

        for (size_t i = 0; i < grid.size() - 1; ++i) {
            if (value >= grid[i] && value <= grid[i + 1]) {
                double weight = (value - grid[i]) / (grid[i + 1] - grid[i]);
                return {static_cast<int>(i), static_cast<int>(i + 1), weight};
            }
        }
    } 

    // Recursive function for multidimensional linear interpolation
    double multilinear_interpolation(const std::vector<int>& lower_indices,
                                      const std::vector<double>& weights,
                                      size_t dim,
                                      double weight_product) const {
        if (dim == dim_) {
            // Base case: compute weighted grid value
            size_t index = 0;
            for (size_t d = 0; d < dim_; ++d) {
                index = index * grid_points_[d].size() + lower_indices[d];
            }
            return grid_values_[index] * weight_product;
        }

        // Recursive case: interpolate in the current dimension
        auto lower_indices_next = lower_indices;
        double lower_value = multilinear_interpolation(lower_indices, weights, dim + 1, weight_product * (1 - weights[dim]));

        lower_indices_next[dim]++;
        double upper_value = multilinear_interpolation(lower_indices_next, weights, dim + 1, weight_product * weights[dim]);

        return lower_value + upper_value;
    } 
};


double getValue2D (double x, double y, const std::string& filename) {

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) { 
        std::cerr << "Error: Could not open the file" << std::endl; // if file cant be opened
        return 1;
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(filename, line)) {
        std::istringstream iss(line); // creates SS of the line its looking at in the file
        std::vector<double> row; 
        double value;
        

        while (iss >> value) {
            row.push_back(value); // chops up SS and pushes each value individually to data
        }
        data.push_back(row);
    }

    inputFile.close();
    
    std::vector<double> xValues;
    std::vector<double> yValues;

    for (size_t i = 1; i < data[0].size(); ++i) {
        xValues.push_back(data[0][i]);
    }

    for (size_t i = 1; i < data.size(); ++i) {
        yValues.push_back(data[i][0]);
    }

    std::vector<double> grid_values;
    for (size_t i = 1; i < data.size(); ++i) {
        for (size_t j = 1; j < data[i].size(); ++j) {
            grid_values.push_back(data[i][j]);
        }
    }

    // Define grid points for interpolation
    std::vector<std::vector<double>> grid_points = {xValues, yValues};

    std::vector<std::vector<double>> grid_points = {
        xValues, 
        yValues
    };

    RegularGridInterpolator interpolator(grid_points, grid_values);
    std::vector<double> point = {x, y};
    double result = interpolator.interpolate(point); // interp = RegularGridInterpolator((yValues, xValues), dataValues) 
    
    return result; 
}

std::optional<size_t> findAboveIndex(std::vector<double>& zValues, double z) {
    for (size_t i = 0; i < zValues.size(); ++i) {
        if (zValues[i] > z) {
            return i;  // return the first index where value > z
        }
    }
    return std::nullopt;  // return null if no value is found
}

std::optional<size_t> findBelowIndex(std::vector<double>& zValues, double z, std::optional<size_t> aboveIndex) {
    if (aboveIndex != std::nullopt) {
        int aboveIndex;
        return (aboveIndex - 1); // return the index below aboveIndex as the point from which the function should interpolate below
    }
    return std::nullopt; // return null if the value of aboveInex is null
}

double getValue3D (double x, double y, double z, const std::string& filename) {

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open the file" << std::endl;
        return 1;
    
    std::vector<std::vector<std::vector<double>>> data;
    std::vector<std::vector<double>> data_2d;
    std::string line;
    
    while (std::getline(filename, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        data_2d.push_back(row);
    }
    }

    inputFile.close();

    std::vector<int> nanIndices;

    // now filter the 2d vector 
    for (size_t i = 0; i < data_2d.size(); ++i) {
        if (std::isnan(data_2d[i][0])) {
            int nancoords[2] = {i,0}; // only checks first column, may need to change this
            nanIndices.push_back(nancoords);
        }
    }

    std::vector<int> jumpIndices;
    jumpIndices.push_back(0); //jumpIndices inital value = 0

    
    for (size_t i = 1; i < nanIndices.size(); ++i) {
        if (nanIndices[i] - nanIndices[i - 1] > 1) {
            jumpIndices.push_back(nanIndices[i]); // jumpIndices contains the indices of not a number values 
        }
    }

    std::vector<std::vector<std::vector<double>>> dataArrays; // create 3d data arrays
    for (size_t i = 0; i < jumpIndices.size() - 1; ++i) {
        double start = jumpIndices[i]; // gets first entry of subarray, NOTE: jump indices will show where the subarrays start and end 
        double end = jumpIndices[i + 1]; // gets last entry to jumpIndices

        std::vector<std::vector<double>> subarray; // define subarray vector
        for (int row = start; row < end; ++row) { 
            subarray.push_back(data_2d[row]); // add the value of data, indexed at the value of the first entry of jump indices
        }
        dataArrays.push_back(subarray); // add subarray to data array
    }
    
    std::vector<std::vector<double>> lastSubarray; 
    for (size_t row = jumpIndices.back(); row < data_2d.size(); ++row) {
        lastSubarray.push_back(data_2d[row]); // last subarray handled seperately for whatever is left
    }
    dataArrays.push_back(lastSubarray); // add last subarray to data array

    std::vector<double> zValues; // define zValues

    for (size_t i = 0; i < dataArrays.size(); i++) { // iterate through dataArrays
        std::vector<std::vector<double>> dataArray = dataArrays[i]; // define data array as one subarray in dataarrays
        zValues.push_back(dataArray[0,1]); // add the z array in the subarray to zvalues
    }

    auto aboveIndex = findAboveIndex(zValues, z); // define aboveIndex from findAboveIndex function written above
    auto belowIndex = findBelowIndex(zValues, z, aboveIndex); // define belowIndex from findBelowIndex function written above


    if (z == zValues[aboveIndex]) {
        belowIndex = aboveIndex; // if provided z is within zValues then the exact layer has been located so no need to interpolates between layers
    }
    
    // Create interpolators for the two closest layers when z is not in zvalues 
    RegularGridInterpolator interpAbove(
        {dataArrays[aboveIndex][3], dataArrays[aboveIndex][2]}, dataArrays[aboveIndex][3]); // interpolates layer above 
    RegularGridInterpolator interpBelow(
        {dataArrays[belowIndex][3], dataArrays[belowIndex][2]}, dataArrays[belowIndex][3]); // interpolates layer below
    
    if (z == zValues[aboveIndex]) {
        return interpAbove.interpolate({x, y}); // if provided z is within zValues then only interpolate 1 layer, if this condition is met then aboveIndex = belowIndex already
    }
    
    double valueBelow = interpBelow.interpolate({x, y});
    double valueAbove = interpAbove.interpolate({x, y});
    
    return ((valueBelow * (zValues[aboveIndex] - z)) + (valueAbove * (z - zValues[belowIndex]))) /
           (zValues[aboveIndex] - zValues[belowIndex]);
}
    

// double check what rocket class is called and what outputs are this time
double getAeroParams(double M, double alpha, double logR, double Cvf, double Cpl, double Ccl, double lfd, double lcd, double lad, double dbd, double L, double D, double sweep, double rcd, double tcd, double spand, double gapd, double led, double ld, double xm, double xml) {

    // getting values to pass into function
    Rocket rocket;
    double [Cvf, Cpl, Ccl, lfd, lcd, lad, dbd, L, D, sweep, rcd, tcd, spand, gapd, led, ld, xm, xml] = Rocket.getGeoParams();

    double Re = std::pow(10.0, logR);
    double beta = std::pow(abs(std::pow(M, 2.0) - 1.0), 0.5);

    if (M >= 0.8 || M <= std::pow(1.36,0.5)) {
        beta = 0.64; // avoid singularities
    }
    if (beta >= 1.0 & beta <= 1.001) {
        beta = 1.001;  // avoid singularities
    }

    // interpolating from ESDU 89008
    double Cna_l = getValue2D(M, lfd, "inviscid.csv");
    double k = getValue3D(lfd, lcd, M, "overflowFactor.csv");
    double CmCn = getValue3D(lfd, lcd, M, "momentRatio.csv");
    double dStarL = 0.001 * getValue2D(M, logR, "displacementThickness.csv");
    double Cf = 0.001 * getValue2D(M, logR, "skinFriction.csv");

    // force and moment coefficient calculations from ESDU 89008
    double delta_CmCn = (Cvf - 0.54) * lfd;
    double delta_Cna_d = 8.0 * dStarL * ld;
    double delta_Cna_f = 4.0 * Cf * ld;
    double Cna = k * Cna_l + delta_Cna_d + delta_Cna_f;
    double delta_Cm0a_d = -4.0 * dStarL * std::pow(ld,2.0);
    double delta_Cm0a_f = -2.0 * Cf * std::pow(ld,2.0);
    double Cm0a = k * Cna_l * (CmCn + delta_CmCn) + delta_Cm0a_d + delta_Cm0a_f;  // about the nose

    // Cnc from ESDU 89014
    double Cnc = getValue2D(M, alpha, "vortexGeneration.csv");

    // convert coefficients to forces and moments using ESDU 89014
    alpha = ((M_PI)/180.0) * alpha; //convert to radians
    double Cn = Cna * sin(alpha) * cos(alpha) + 4.0 / (M_PI) * ld * Cpl * Cnc;
    double Cm0 = Cm0a * sin(alpha) * cos(alpha) - 2.0 / (M_PI) * std::pow(ld,2.0) * Cpl * Ccl * Cnc;
    double Cma = Cna * (xml * ld) + Cm0a;
    double Cm = Cn * (xml * ld) + Cm0;  // pitching moment about midpoint
    double delta_Cna;
    // modifications due to boattail, from ESDU 87033
    double A = -1.3 * lad + 6.35 * lad - 7.85;
    if (M >= 0.0 & M < 1.0) {
        delta_Cna = -2.0 * (1.0 - std::pow(dbd,2));
    } else {
        delta_Cna = -(0.6 + 1.4 * exp(A * std::pow((M - 1.0),0.8)) * (1.0 - std::pow(dbd,2.0)));
    }
    double delta_Cdc;
    double delta_Cn;
    delta_Cdc = -(2.0 + tanh(1.5 * M * sin(alpha) - 2.4)) * lad * (1.0 - dbd);
    delta_Cn = delta_Cna * sin(alpha) * cos(alpha) * (1 - (std::pow(sin(alpha),0.6))) + delta_Cdc * (std::pow(sin(alpha),2));
    Cn += delta_Cn;
    // we don't modify Cm as the fins affect it instead (trust ESDU)
    double xcp;
    if (alpha == 0.0) {  // avoid division by zero
        Cm = 0.0;
        xcp = -Cm0a / Cna;
    } else {
        xcp = xml * ld - Cm / Cn;
    }

    // get fin parameters (ESDU 91004 Appendix)
    double lam = tcd / rcd;
    double sf = 0.5 * (tcd + rcd) * (spand - gapd);  // * D^2
    double AR = std::pow((spand - gapd), (2.0 / sf));
    double back = AR * tan((M_PI / 180.0) * sweep);  // distance swept back at mid-chord
    double betaAR = std::pow(beta,0.5) * AR;

    double Cna_fin;
    double xOverC;

    // get fin lift-curve slope etc. from ESDU 70011/70012
    if (M <= 1) {
        Cna_fin = AR * getValue3D(back, betaAR, lam, "finSubsonic.csv");
        xOverC = getValue3D(lam, betaAR, back, "finCentreSubsonic.csv");
    } else {
        Cna_fin = AR * getValue3D(back, betaAR, lam, "finSupersonic.csv");
        xOverC = getValue3D(lam, betaAR - back, back, "finCentreSupersonic.csv");
    }

    double xOverD = xOverC * rcd / (2.0 * (1.0 + lam + std::pow(lam,2.0))) * (3.0 * (1.0 + lam));  // TODO: check this?
    double xfl = (led + xOverD) / ld;  // distance from nose to x_ac
    

    // fin interference (this bit is really annoying - see ESDU S.01.03.01)
    double r = 0.5 * D;
    double s = spand * D / 2.0;
    double Kwb = (
        (2.0 /(M_PI))
        * (
            (1 + std::pow(r,4.0) / std::pow(s,4.0)) * (0.5 * atan(0.5 * (s / r - r / s)) +(M_PI / 4.0)
            - (std::pow(r,2.0) / std::pow(s,2.0)) * ((s / r - r / s) + 2.0 * atan(r / s))
        ))
        / std::pow((1.0 - r / s),2.0)
    );
    // the following is used to determine the shocks on the fin
    double shockParam = AR * std::pow(beta,0.5) * (1.0 + lam) * (tan(((M_PI)/180.0)*(sweep)) + 1.0);
    double Kbw;
    if (shockParam < 4.0) {
        // this is the subsonic case, there is only one equation
        Kbw = std::pow((1.0 + r / s),2.0) - Kwb;
    } else {
        // define a load of dimensionless parameters to make the equation simpler
        double pB = std::pow(beta,-0.5) / tan(((M_PI)/180.0)*(sweep));  // B
        double pD = 2.0 * r * std::pow(beta,0.5) / (rcd * D);  // D
        double pP = (ld - rcd - led) * D / (2.0 * rcd * std::pow(beta,0.5)); // P
        double pP = std::min(pP, 1.0);  // P is limited to 1
        double pR = pP + 1.0 / pD;  // R
        // subsonic LE (B < 1)
        if (pB < 1.0) {
            if (pR < 1.0) {
                Kbw = ((16.0 * std::pow(pB,0.5) * pD) / (M_PI * (pB + 1.0))) * (
                    ((std::pow(pB,1.5) / (std::pow(pD,2.0) * (1.0 + pB))) * (std::pow(((pB + (1.0 + pB) * pP * pD) / pB),0.5)) - 2.0)
                    - (pB / (1.0 + pB)) * std::pow(pD,-0.5) * std::pow((pB * pR + pP),1.5)
                    + pB * (1.0 + pB) * std::pow(pR,2.0) * std::pow(atan(1.0 / (pD * (pB * pR + pP))),0.5)
                );
            } else {  
                Kbw = ((16.0 * std::pow(pB,0.5) * pD) / (M_PI * (pB + 1.0))) * (
                    ((std::pow(pB,1.5) / (std::pow(pD,2.0) * (1.0 + pB))) * (std::pow(pB + (1.0 + pB) * pP * pD / pB,0.5) - 2.0))
                    - (pB / (1.0 + pB)) * std::pow(pD,-0.5) * std::pow((pB * pR + pP),1.5)
                    + pB * (1.0 + pB) * std::pow(pR,2.0) * std::pow(atan(1.0 / (pD * (pB * pR + pP))),0.5)
                ) + ((16.0 * std::pow(pB,0.5) * pD) / (M_PI * (pB + 1.0))) * (
                    ((pB * pR + 1.0) * ((pR - 1.0) * std::pow((pB * pR + 1.0),0.5)))
                    - (pB + 1.0) / (std::pow(pB,0.5)) * atanh((pB * pR - pB)) / std::pow((pB * pR + 1.0),0.5) 
                    - pB * (1.0 + pB) * std::pow(pR,2.0) * std::pow(atan((pR - 1.0) / (pB * pR + 1.0)),0.5)
                );
            }
        } else {  // supersonic LE (B > 1)
            if (pR < 1.0) {
                Kbw = (
                    (8.0 * pD)
                    / (M_PI * std::pow((std::pow(pB,2.0) - 1.0),0.5))
                    * (
                        (-pB / (1.0 + pB)) * std::pow((pB * pR + pP),2.0) * acos((pR + pB * pP) / (pB * pR + pP))
                        + ((pB * std::pow((std::pow(pB,2.0) - 1.0),0.5)) / (std::pow(pD,2.0) * (1.0 + pB))) * (std::pow((1.0 + 2.0 * pP * pD),0.5) - 1.0)
                        - (std::pow(pB,2.0)) / (std::pow(pD,2.0) * (1.0 + pB)) * acos(1.0 / pB)
                        + (pB * std::pow(pR,2.0) * std::pow((std::pow(pB,2.0) - 1.0),0.5) * acos(pP / pR))
                    )
                );
            } else {
                Kbw = (
                    (8.0 * pD)
                    / (M_PI * std::pow((std::pow(pB,2.0) - 1.0),0.5)))
                    * (
                        (-pB / (1.0 + pB)) * std::pow((pB * pR + pP),2.0) * acos((pR + pB * pP) / (pB * pR + pP))
                        + ((pB * (std::pow((std::pow(pB,2.0) - 1.0),0.5)) / (std::pow(pD,2.0) * (1.0 + pB))) * (std::pow((1.0 + 2.0 * pP * pD),0.5) - 1.0)
                        - (std::pow(pB,2.0)) / (std::pow(pD,2.0) * (1.0 + pB)) * acos(1.0 / pB)
                        + (pB * std::pow(pR,2.0) * std::pow((std::pow(pB,2.0) - 1.0),0.5) * acos(pP / pR))
                    )
                ) + ((8.0 * pD) / (M_PI * (std::pow(std::pow(pB,2.0) - 1.0),0.5))) * (
                    std::pow((pB * pR + 1.0),2.0) * acos((pR + pB) / (pB * pR + 1.0))
                    - std::pow((std::pow(pB,2.0) - 1.0),0.5) * acosh(pR)  //again need to ask about cosh
                    + pB * std::pow(pR,2.0) * std::pow((std::pow(pB,2.0) - 1.0),0.5) * (asin(1.0 / pR) -(M_PI / 2.0)
                ));
            }
        }
    }

    // convert to the correct form
    Kbw /= beta * Cna_fin * (1.0 + lam) * (s / r - 1.0);

    // now we need pitch damping derivatives, from ESDU 91004
    double xcl = xml;  // TODO: distance of volume centroid from the nose tip
    // F1 = getValue2D(M, xml, "f1.csv")  # TODO: I literally made the subsonic numbers up
    // F2 = (1.045 * ld**2 - 0.438 * ld + 8.726) / (ld**2 - 1.009 * ld + 12.71)
    double MqPlusMwdot = -Cna * std::pow((1.0 - xml),2) * std::pow(ld,2.0);  // use F1*F2-xml if we consider boattail effects
    double Mwdot = Cma * ((Cvf * (xcl - xml) * ld) / ((1.0 - xml) * std::pow(dbd,2.0) - Cvf));

    // incorporate the fin effects
    double Sf = (tcd + rcd) * (spand - gapd) * std::pow(D,2.0) / 4.0;
    double Sref =M_PI * std::pow(D,2.0) / 4.0;
    double Mq = MqPlusMwdot - Mwdot - Cna_fin * (Kwb + Kbw) * (Sf / Sref) * std::pow((xfl - xml),2.0) * std::pow(ld,2.0);

    
    // Cv is the ratio of the volumes of the rocket and its enclosing cylinder
    // Cs is the ratio of the surface area of the rocket to the surface area of its enclosing cylinder
    double Cva = 1.0 / 3.0 * (1.0 + dbd + std::pow(dbd,2.0));
    double Cv = lcd / (ld + lad) + Cva * (lad / (ld + lad)) + Cvf * (lfd / (ld + lad));
    double Csa = 0.5 * (1 + dbd) * std::pow((1.0 + 0.25 * std::pow(((1.0 - dbd) / lad),2.0)),0.5);
    double Csf = (0.2642 * std::pow(lfd,-2) + 0.6343 * std::pow(lfd,-1) + 2.214) / (std::pow(lfd,-1) + 3.402);
    double Cs = lcd / (ld + lad) + Csa * (lad / (ld + lad)) + Csf * (lfd / (ld + lad));
    double Fm1;

    if (M == 0.0) {
        Fm1 = 0.1011;
    } else {
        Fm1 = 0.18 * std::pow(M,2) / std::pow((atan(0.4219 * M)),2.0);
    }
    double Fm2 = std::pow((1.0 + 0.178 * std::pow(M,2)),-0.702) / Fm1;
    double B = 2.62105 - 0.0042167 * log10(Fm2 * Re);
    double Cf0 = 0.455 / (Fm1 * std::pow((log10(Fm2 * Re)),B));
    double lfOverL = lfd / ld;

    // skin friction
    double xtrOverL = 0.95 * lfOverL;  // transition to turbulent flow, distance from nose divided by rocket length
    double F1 = 41.1463 * std::pow(Re,-0.377849);
    double g = 0.71916 - 0.0164 * log10(Re);
    double h = 0.66584 + 0.02307 * log10(Re);
    double F2 = 1.1669 * std::pow(log10(Re),-3.0336) - 0.001487;
    double Cf = Cf0 * ((1.0 - xtrOverL + F1 * std::pow(std::pow(xtrOverL,g)),h) - F2);

    // other conversion factors
    double Ktr = 1.0 + 0.36 * xtrOverL - 3.3 * std::pow(xtrOverL,3.0);
    double b = lfOverL * (1.0 - Cvf);
    double Fm = 1.5 * std::pow(M,2.0) * (1.0 + 1.5 * std::pow(M,4.0));
    double Fb;

    if (b < 0.03) {
        Fb = 0.0;
    } else if (b < 0.15) {
        Fb = 0.0924 / b + 0.725 * b + 12.2 * std::pow(b,2.0);
    } else {
        Fb = 1.0;
    }

    double Km = 1.0 + Fm * Fb * std::pow((1.0 / ld),2.0);

    // final calculations
    double CdV_fins = 0.0;
    double CdV_nose = Cf * Ktr * Km * 3.764 * (std::pow((1.0 / ld),(-1.0 / 3.0)) + 1.75 * std::pow((1.0 / ld),(7.0 / 6.0)) + 3.48 * std::pow((1.0 / ld),(8.0 / 3.0)));
    double CdV = CdV_nose + CdV_fins; // Cdv seems to be undefined ask ben 
    double Cdp = CdV * (std::pow(Cv,(2.0 / 3.0)) / (2.0 * std::pow((2.0 *(M_PI) * ld),(1.0 / 3.0)) * Cs));

    // look at B.S.02.03.01 (very important!)

    // TODO: look at ESDU 76033/78041/79022 to add base drag
    double boattailAngle = (180.0/(M_PI)) * (atan2(1.0 - dbd, 2.0)); // what is arctan2
    double Cdb;
    double Cd_beta;


    if (M <= 0.8) {
        // Subsonic base drag - ESDU 76033
        Cdb = getValue2D(dbd, boattailAngle, "baseDragSubsonic.csv");
        Cd_beta = getValue3D(dbd, boattailAngle, M, "boattailDragSubsonic.csv");
    } else if (M <= 1.3) {
        // Transonic base drag - ESDU 78041
        Cdb = getValue3D(dbd, boattailAngle, M, "baseDragTransonic.csv");
        Cd_beta = getValue3D(dbd, boattailAngle, M, "boattailDragTransonic.csv");
    } else {
        // Supersonic base drag - ESDU 79022
        Cdb = getValue3D(std::pow(dbd,2.0), boattailAngle, M, "baseDragSupersonic.csv");
        // ESDU B.S.02.03.02 - boattail drag coefficient (wave drag)
        Cd_beta = std::pow((D / (2.0 * lad)),2.0) * getValue3D(std::pow(dbd,2.0), boattailAngle, M, "boattailDragSupersonic.csv");
    }

    double F = getValue2D(M, alpha, "angleDrag.csv");  // angle of attack effect on base drag
    // TODO: Use ESDU 02012 for the effect of the jet - ignore this until we can get exhaust temperature and pressure

    double Cdw = 0;  // wave drag - use ESDU B.S.02.03.01
    double Cdwv = 0;  // viscous form drag?
    double Cd = Cdp + Cdw + Cdwv + F * (Cdb + Cd_beta) + Cf;

    return Cn, Cm, xcp, Mq, Cd, Cdp, Cdw, Cdwv, F * (Cdb + Cd_beta), Cf;
}








