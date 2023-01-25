# Plot the slope gradient of f(x) = ax^2 
# https://www.mathworks.com/matlabcentral/answers/281511-how-do-i-plot-the-plot-a-line-using-slope-and-one-x-y-coordinate-on-matlab

x = -3:0.25:3;

m = 16;
y = m*(x.^2);
h = plot(x, y);
hold on;

x1 = -3:1;
y1 = 16;
y = -(2*m*x1 + y1);
h = plot(x1, y);

grid on;
title ("s(t) = 16t^2");
xlabel ("t (seconds)");
ylabel ("s(t)");

print -djpg "04_slope.jpg"
waitfor(h);