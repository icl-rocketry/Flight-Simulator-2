clear all
clc

% read the data from the file
data = csvread("flight_data.txt");

% time, position, velocity, acceleration, mass
t = data(:, 1);
z = data(:, 2);
v = data(:, 3);
a = data(:, 4);
m = data(:, 5);

% do 2x2 subplots
subplot(2, 2, 1);
plot(t, z);
title("Altitude")
xlabel("Time (s)")
ylabel("Altitude (m)")

subplot(2, 2, 2);
plot(t, v);
title("Velocity")
xlabel("Time (s)")
ylabel("Velocity (m/s)")

subplot(2, 2, 3);
plot(t, a)
title("Acceleration")
xlabel("Time (s)")
ylabel("Acceleration (m/s^2)")

subplot(2, 2, 4);
plot(t, m)
title("Mass")
xlabel("Time (s)")
ylabel("Mass (kg)")
