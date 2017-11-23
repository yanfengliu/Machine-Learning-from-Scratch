close all;
clear;
rng(1);

% number of runs of the algorithm
NUM_RUNS = 5;
% number of iterations
NUM_ITERATIONS = 100;
% number of neurons
neuron_array = [5];
% number of particles in the swarm
NUM_PARTICLES = 25;
% inertia constant
w = 0.6;
% set speed limit factor as ratio to range of search space
SPEED_LIMIT = 0.6;
% search space limit
LIMIT = 5;
% acceleration constants
c1 = 1.5;
c2 = 2;

for k = 1:length(neuron_array)
    NUM_NEURONS = neuron_array(k);
    MSE_array = zeros(1, NUM_RUNS);
    converge_array = zeros(1, NUM_RUNS);
    for r = 1:NUM_RUNS
        % Initialize particles
        swarm.pos = rand(3 * NUM_NEURONS + 1, NUM_PARTICLES) * (2*LIMIT) - LIMIT;
        swarm.v = rand(3 * NUM_NEURONS + 1, NUM_PARTICLES) * (2*LIMIT) - LIMIT;
        swarm.z = zeros(1, NUM_PARTICLES);
        for i = 1:NUM_PARTICLES
            swarm.z(i) = get_fitness(swarm.pos(:, i), NUM_NEURONS);
        end
        swarm.z_pbest = swarm.z;
        swarm.z_gbest = min(swarm.z_pbest);
        swarm.pos_pbest = swarm.pos;
        swarm.pos_gbest = swarm.pos(:, swarm.z == min(swarm.z));
        swarm.converge = 0;
        
        for i = 1:NUM_ITERATIONS
            for j = 1:NUM_PARTICLES
                % move w1 and w2
                swarm.v(:, j) = w * swarm.v(:, j) ...
                    + c1 * rand(3*NUM_NEURONS+1, 1) .* (swarm.pos_pbest(:, j) - swarm.pos(:, j))...
                    + c2 * rand(3*NUM_NEURONS+1, 1) .* (swarm.pos_gbest - swarm.pos(:, j));
                
                % apply speed limit
                swarm.v(:, j) = min(swarm.v(:, j), SPEED_LIMIT);
                swarm.v(:, j) = max(swarm.v(:, j), -SPEED_LIMIT);
                
                % apply speed
                swarm.pos(:, j) = swarm.pos(:, j) + swarm.v(:, j);
                
                % bounce back particles leaving search space
                out_idx = ((swarm.pos(:, j) > LIMIT) | (swarm.pos(:, j) < -LIMIT));
                swarm.v(out_idx, j) = -swarm.v(out_idx, j);
                
                % make sure parameters don't leave search space
                swarm.pos(:, j) = min(swarm.pos(:, j), LIMIT);
                swarm.pos(:, j) = max(swarm.pos(:, j), -LIMIT);
                
                % calculate new z
                swarm.z(j) = get_fitness(swarm.pos(:, j), NUM_NEURONS);
                
                % check for new particle best
                if swarm.z(j) < swarm.z_pbest(j)
                    swarm.pos_pbest(:, j) = swarm.pos(:, j);
                    swarm.z_pbest(j) = swarm.z(j);
                    % check for global best
                    if swarm.z(j) < swarm.z_gbest
                        swarm.z_gbest = swarm.z(j);
                        swarm.pos_gbest = swarm.pos(:, j);
                        swarm.converge = i;
                    end
                end
            end
        end
        MSE_array(r) = swarm.z_gbest;
        converge_array(r) = swarm.converge;
    end
    fprintf('%d & %e \\\\ \n', NUM_NEURONS, mean(MSE_array));
end


w1 = swarm.pos_gbest(1:NUM_NEURONS);
w2 = swarm.pos_gbest((NUM_NEURONS+1):(2*NUM_NEURONS));
b1 = swarm.pos_gbest((2*NUM_NEURONS+1):(3*NUM_NEURONS));
b2 = swarm.pos_gbest(3*NUM_NEURONS+1);

x = rand(1, 1000) * 2 - 1;
y_pred = zeros(1, 1000);
y_true = y_pred;
for i = 1:1000
    z1 = w1 * x(i) + b1;
    a1 = sigmoid(z1);
    z2 = dot(w2, a1) + b2;
    y_pred(i) = z2;
    y_true(i) = 2 * x(i)^2 + 1;
end
figure;
hold on;
scatter(x, y_pred, 5, 'b');
scatter(x, y_true, 5, 'r');

function fitness = get_fitness(pos, n)
    w1 = pos(1:n);
    w2 = pos((n+1):(2*n));
    b1 = pos((2*n+1):(3*n));
    b2 = pos(3*n+1);
    x = rand(1, 1000) * 2 - 1;
    y_pred = zeros(1, 1000);
    y_true = y_pred;
    for i = 1:1000
        z1 = w1 * x(i) + b1;
        a1 = sigmoid(z1);
        z2 = dot(w2, a1) + b2;
        y_pred(i) = z2;
        y_true(i) = 2 * x(i)^2 + 1;
    end
    fitness = mean((y_pred - y_true).^2);
end

function [y] = sigmoid(x)
    y = 1./(1+exp(-x));
end