#
# Plot the slope gradient for f(x) = ax^2
# Derivative: f'(x) = 2ax 
#

x = -3:0.25:3;

m = 16;
y = m*(x.^2);
h = plot(x, y);
hold on;

x1 = -2:1;
y1 = 16;
y = -(2*m*x1 + y1);
h = plot(x1, y);

grid on;
title ("s(t) = 16t^2");
xlabel ("t (seconds)");
ylabel ("s(t)");

img = '04_slope.jpg'
print -djpg '04_slope.jpg'
copyfile '04_slope.jpg' '../../../../../../html/lib/images/ml/04_slope.jpg'

waitfor(h);