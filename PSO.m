close all;
clear;

rng(100);

% Important values

% number of iterations
ITERATIONS = 100;
% number of runs
NUM_RUNS = 100;
% number of particles
NUM_PARTICLES = 50;
% inertia constant
w = 0.3;
% set speed limit factor as ratio to range of search space
SPEED_LIMIT = 0.6;
% acceleration constants
c1 = 2;
c2 = 2;

colors = rand(20, 3);

% Update particles

x_array = zeros(1, NUM_RUNS);
y_array = zeros(1, NUM_RUNS);
z_array = zeros(1, NUM_RUNS);
itr_array = zeros(1, NUM_RUNS);

for k = 1:NUM_RUNS
    % Initialize particles
    swarm.x = rand(1, NUM_PARTICLES) * 10 - 5;
    swarm.y = rand(1, NUM_PARTICLES) * 10 - 5;
    swarm.vx = rand(1, NUM_PARTICLES) * 2 - 1;
    swarm.vy = rand(1, NUM_PARTICLES) * 2 - 1;
    swarm.z = zeros(1, NUM_PARTICLES);
    for i = 1:NUM_PARTICLES
        swarm.z(i) = get_z_value(swarm.x(i), swarm.y(i));
    end
    swarm.z_pbest = swarm.z;
    swarm.z_gbest = min(swarm.z_pbest);
    swarm.x_pbest = swarm.x;
    swarm.x_gbest = swarm.x(swarm.z == min(swarm.z));
    swarm.y_pbest = swarm.y;
    swarm.y_gbest = swarm.y(swarm.z == min(swarm.z));
    swarm.converge = 0;
    
    for i = 1:ITERATIONS
        for j = 1:NUM_PARTICLES
            % move x and y
            swarm.vx(j) = w * swarm.vx(j) + c1 * rand * (swarm.x_pbest(j) - ...
                swarm.x(j)) + c2 * rand * (swarm.x_gbest - swarm.x(j));
            swarm.vy(j) = w * swarm.vy(j) + c1 * rand * (swarm.y_pbest(j) - ...
                swarm.y(j)) + c2 * rand * (swarm.y_gbest - swarm.y(j));
            
            % apply speed limit
            swarm.vx(j) = min(swarm.vx(j), SPEED_LIMIT);
            swarm.vx(j) = max(swarm.vx(j), -SPEED_LIMIT);
            swarm.vy(j) = min(swarm.vy(j), SPEED_LIMIT);
            swarm.vy(j) = max(swarm.vy(j), -SPEED_LIMIT);
            
            % apply speed
            swarm.x(j) = swarm.x(j) + swarm.vx(j);
            swarm.y(j) = swarm.y(j) + swarm.vy(j);
            
            % make sure it doesn't leave the search space
            swarm.x(j) = max(-5, swarm.x(j));
            swarm.x(j) = min(5, swarm.x(j));
            swarm.y(j) = max(-5, swarm.y(j));
            swarm.y(j) = min(5, swarm.y(j));
            
            % calculate new z
            swarm.z(j) = get_z_value(swarm.x(j), swarm.y(j));
            
            % check for new particle best
            if swarm.z_pbest(j) < swarm.z(j)
                swarm.x_pbest(j) = swarm.x(j);
                swarm.y_pbest(j) = swarm.y(j);
                swarm.z_pbest(j) = swarm.z(j);
            end
        end
        %         hold off;
        % check for new global best
        [z_min, idx] = min(swarm.z);
        if z_min < swarm.z_gbest
            swarm.z_gbest = z_min;
            swarm.x_gbest = swarm.x(idx);
            swarm.y_gbest = swarm.y(idx);
            swarm.converge = i;
        end
    end
    
    x_array(k) = swarm.x_gbest;
    y_array(k) = swarm.y_gbest;
    z_array(k) = swarm.z_gbest;
    itr_array(k) = swarm.converge;
end
scatter(x_array, y_array);

x_array_1 = x_array(x_array > 0);
y_array_1 = y_array(x_array > 0);
mean(x_array_1)
mean(y_array_1)

x_array_2 = x_array(x_array < 0);
y_array_2 = y_array(x_array < 0);
mean(x_array_2)
mean(y_array_2)

% mesh plot
X = -1:0.01:1;
Y = -1:0.01:1;
Z = zeros(length(X), length(Y));
for i = 1:length(X)
    for j = 1:length(Y)
        Z(i, j) = get_z_value(X(i), Y(j));
    end
end

figure;
mesh(X, Y, Z);
[x, y] = find(Z == min(min(Z)));
fprintf('Z global best from meshgrid = %f at (%f, %f)\n', min(min(Z)), X(x(1)), Y(y(1)));
fprintf('Z global best from meshgrid = %f at (%f, %f)\n', min(min(Z)), X(x(2)), Y(y(2)));

function value = get_z_value(x, y)
value = (4 - 2.1 * x^2 + (x^4) / 3) * x^2 + x * y + (-4 + 4 * y^2) * y^2;
end
