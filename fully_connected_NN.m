close all;
clear;
% set seed for repeatable random numbers
rng(189);
% network settings
NUM_NEURONS = 5;
LEARNING_RATE = 0.03;
MOMENTUM_WEIGHT = 0.9;
EPOCHS = 10;
% initialize network
net = initialize_net(NUM_NEURONS, LEARNING_RATE, MOMENTUM_WEIGHT);
% train the network
x = rand(1, 1000) * 2 - 1;
y = 2 * x.^2 + 1;
[loss_array, net] = train(net, x, y, EPOCHS);
% test the network
x_test = rand(1, 1000) * 2 - 1;
y_pred = predict(net, x_test);
y_test = 2 * x_test.^2 + 1;
% plot loss vs epoch
figure;
plot(loss_array, 'LineWidth', 5);
title_string = sprintf('Loss vs Epoch, Learning Rate %.2f, Momentum %.2f', ...
    LEARNING_RATE, MOMENTUM_WEIGHT);
title(title_string);
fprintf('Learning rate: %.2f, Momentum: %.2f, Minimum loss: %e, Avg loss: %.5f\n', ...
    LEARNING_RATE, MOMENTUM_WEIGHT, min(loss_array), mean(loss_array));
% plot prediction vs truth
figure;
scatter(x_test, y_test, 5, 'b');
hold on;
scatter(x_test, y_pred, 5, 'r');
title_string = sprintf('Prediction vs Truth, Learning Rate %.2f, Momentum %.2f', ...
    LEARNING_RATE, MOMENTUM_WEIGHT);
title(title_string);
legend('test truth', 'predicted value');

function net = initialize_net(NUM_NEURONS, LEARNING_RATE, MOMENTUM_WEIGHT)
    net.learning_rate = LEARNING_RATE;
    net.momentum_weight = MOMENTUM_WEIGHT;
    net.W1 = randn(1, NUM_NEURONS) * 0.2;
    net.b1 = zeros(1, NUM_NEURONS);
    net.W2 = randn(1, NUM_NEURONS) * 0.2;
    net.b2 = 1;
    net.v_dW1 = zeros(size(net.W1));
    net.v_db1 = zeros(size(net.b1));
    net.v_dW2 = zeros(size(net.W2));
    net.v_db2 = 0;
end

function y = predict(net, x)
    y = zeros(size(x));
    for i = 1:length(x)
        % forward propagation
        Z1 = net.W1 * x(i) + net.b1;
        A1 = sigmoid(Z1);
        Z2 = dot(net.W2, A1) + net.b2;
        y(i) = Z2;
    end
end

function [loss_array, net] = train(net, x, y, EPOCHS)
    loss_array = zeros(1, EPOCHS);
    for i = 1:EPOCHS
        for j = 1:length(x)
            % forward propagation
            Z1 = net.W1 * x(j) + net.b1;
            A1 = sigmoid(Z1);
            Z2 = net.W2 * A1' + net.b2;
            loss = 0.5 * mean((Z2 - y(j)).^2);
            % back propagation using chain rule
            dZ2 = Z2 - y(j);
            dW2 = dZ2 * A1;
            db2 = dZ2;
            dA1 = net.W2' * dZ2;
            dZ1 = dA1' .* sigmoid_grad(Z1);
            dW1 = dZ1 * x(j);
            db1 = dZ1;
            % velocity update with momentum
            net.v_dW2 = net.momentum_weight * net.v_dW2 - net.learning_rate * dW2;
            net.v_db2 = net.momentum_weight * net.v_db2 - net.learning_rate * db2;
            net.v_dW1 = net.momentum_weight * net.v_dW1 - net.learning_rate * dW1;
            net.v_db1 = net.momentum_weight * net.v_db1 - net.learning_rate * db1;
            % update parameters using velocity
            net.W2 = net.W2 + net.v_dW2;
            net.b2 = net.b2 + net.v_db2;
            net.W1 = net.W1 + net.v_dW1;
            net.b1 = net.b1 + net.v_db1;
        end
        % save loss into array
        loss_array(i) = loss;
    end
end

function [y] = sigmoid(x)
    y = 1./(1+exp(-x));
end

function [y] = sigmoid_grad(x)
    temp = sigmoid(x);
    y = temp .* (1-temp);
end