%% Incrementer circuit complexity

N = [1 2 3 4 5 6 7 8 9 10];
% optimial for zeroed ancilla
inc = [1 2 41 83 105 140 196 232 265 304];
% general circuit
ginc = [1 116 210 338 451 559 700 780 895 1032];
% sato paper
sinc = [1 2 23 61 193 374 806 1590 3458 7268];


[N_fit, inc_fit, inc_label] = add_best_fit(N, inc);
[gN_fit, ginc_fit, ginc_label] = add_best_fit(N, ginc);

figure()
hold on

% Plot the fitted line
plot(N, inc,'LineWidth',2);
plot(N_fit, inc_fit, 'r-', 'DisplayName', 'Fitted Line');
text(max(N_fit) - 3, min(inc_fit) - 200, inc_label, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

plot(N, ginc,'LineWidth',2);
plot(gN_fit, ginc_fit, 'r-', 'DisplayName', 'Fitted Line');
text(max(gN_fit) - 7, min(ginc_fit) + 600, ginc_label, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

plot(N, sinc, 'LineWidth',2);

% title("Circuit Complexity for Incrementer")
legend('Simplified optimizer','','General Optimizer','','Sato et. al')

xlabel('Number of Qubits')
ylabel('Circuit Depth')


function [x_fit, y_fit, equation_str] = add_best_fit(x,y)

% Perform best fit on a line (1st degree polynomial)
coefficients = polyfit(x, y, 1);
x_fit = linspace(min(x), max(x), 100);
y_fit = polyval(coefficients, x_fit);
equation_str = sprintf('y = %.2fx + %.2f', coefficients(1), coefficients(2));

end