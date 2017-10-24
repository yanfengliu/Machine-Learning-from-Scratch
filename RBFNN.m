close all;
clear;
rng(35);
% network settings
% num_list = [2:20, 25, 30];
num_list = [19];

for L = 1:length(num_list)
    NUM_CENTERS = num_list(L);
    % initialize training data
    X = rand(1, 1000) * 2 - 1;

    % run k-means on X cluster to get centers
    [net.centers, net.assignment] = k_means(X, NUM_CENTERS);

    % 1st of the 3 methods to decide widths
    d_max = max(net.centers) - min(net.centers);
    net.widths = zeros(1, NUM_CENTERS) + d_max/sqrt(2*NUM_CENTERS);

    % 2nd of the 3 methods to decide widths
%     r = 2;
%     for i = 1:NUM_CENTERS
%         distances = net.centers(i) - net.centers;
%         distances = distances(distances ~= 0);
%         distances = abs(distances);
%         net.widths(i) = r * min(distances);
%     end

    % 3rd of the 3 methods to decide widths
%     p = 2;
%     for i = 1:NUM_CENTERS
%         distances = net.centers(i) - net.centers;
%         distances = distances(distances ~= 0);
%         distances = sort(abs(distances));
%         distances = distances(1:p);
%         net.widths(i) = sqrt((1/p) * sum(distances.^2));
%     end

    % get linear weights by pseudoinverse method
    y = X.^3 + 2*X.^2 + 0.5*X + 1;
    a = zeros(length(X), NUM_CENTERS);
    for i = 1:length(X)
        for j = 1:NUM_CENTERS
            a(i, j) = gaussian(X(i), net.centers(j), net.widths(j));
        end
    end
    net.b = 1;
    net.w = inv(a' * a) * a' * (y-net.b)';


    % test the network
    X_test = -1:0.01:1;
    y_test = X_test.^3 + 2*X_test.^2 + 0.5*X_test + 1;
    y_test = y_test';
    y_pred = zeros(length(y_test), 1);

    for i = 1:length(y_test)
        a = zeros(1, NUM_CENTERS);
        for j = 1:NUM_CENTERS
            a(j) = gaussian(X_test(i), net.centers(j), net.widths(j));
        end
        y_pred(i) = a * net.w + net.b;
    end
    MSE = 0.5 * mean((y_pred - y_test).^2);
    figure; hold on;
    scatter(X_test, y_test, 5, 'r', 'filled');
    scatter(X_test, y_pred, 5, 'b', 'filled');
    hold off;
    legend('y real value', 'y predicted value');
    title_string = sprintf('Real vs predicted y value on x = [-1, 1], centers = %d\n', NUM_CENTERS);
    title(title_string);
    fprintf('Number of centers = %d, MSE is %e\n', NUM_CENTERS, MSE);

    % test the network when x is a function of time
    SAMPLING_RATE = 10000;
    t = 0:(1/SAMPLING_RATE):0.5;
    X_test = sin(20 * pi * t);
    y_test = X_test.^3 + 2*X_test.^2 + 0.5*X_test + 1;
    y_pred = zeros(1, length(y_test));

    for i = 1:length(y_test)
        a = zeros(1, NUM_CENTERS);
        for j = 1:NUM_CENTERS
            a(j) = gaussian(X_test(i), net.centers(j), net.widths(j));
        end
        y_pred(i) = a * net.w + net.b;
    end
    MSE = 0.5 * mean((y_pred - y_test).^2);
    figure; hold on;
    scatter(t, y_test, 5, 'r', 'filled');
    scatter(t, y_pred, 5, 'b', 'filled');
    hold off;
    legend('y real value', 'y predicted value');
    title_string = sprintf('Real vs predicted y value on t = [0, 0.5], sampling rate = %d Hz\n', SAMPLING_RATE);
    title(title_string);
    fprintf('When x is a function of t, MSE is %e\n', MSE);

    % plot k-means clustering results
    figure; hold on;
    for i = 1:NUM_CENTERS
        temp = X(net.assignment == i);
        c = rand(1, 3);
        scatter(temp, zeros(size(temp)), 20, c, 'filled');
        scatter(net.centers(i), 0, 100, c, 'filled');
    end
end
%% utility functions 
function [centers, assignment] = k_means(X, k)
    n = length(X);
    index = round(rand(1, k) * n);
    C0 = X(index);
    assignment0 = get_assignment(C0, X);
    C1 = k_means_update(assignment0, C0, X);
    assignment1 = get_assignment(C1, X);
    while (sum(assignment0 == assignment1) ~= n)
        C0 = C1;
        assignment0 = assignment1;
        C1 = k_means_update(assignment0, C0, X);
        assignment1 = get_assignment(C1, X);
    end
    centers = C0;
    assignment = assignment0;
end

function assignment = get_assignment(C, X)
    m = length(C);
    n = length(X);
    assignment = zeros(1, n);
    for i = 1:n
        d_min = 1e10;
        for j = 1:m
            d = abs((X(i) - C(j)));
            if (d < d_min)
                d_min = d;
                assignment(i) = j;
            end
        end
    end
end

function C1 = k_means_update(assignment0, C0, X)
    m = length(C0);
    C1 = zeros(size(C0));
    for i = 1:m
        C1(i) = mean(X(assignment0 == i));
    end
end

function y = gaussian(x, center, width)
    y = exp(-((x-center)./width).^2);
end