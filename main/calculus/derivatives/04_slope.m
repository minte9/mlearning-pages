#
# Plot the slope gradient
#
# Function:     s = ax^2
# Derivative:   s' = 2ax 
# Gradient:     f(x) = mx + b
# Coeficient:   m (or slope of the line)
# Intercetp:    b (where the line crosses y-axis)
#

a = 16;

# Plot function line
X = -5:0.1:5;
Y = a*(X.^2);
plot(X, Y);
hold on;

# Plot points and gradients
for x=2:5;
    y = a*(x.^2);
    plot(x, y, 'x', 'Color', 'red');

    m = 2*a*x; # slope coeficient
    b = y - m*x; # intercept in y = mx + b
    X = x:x+2;
    t = num2str(x);
    plot(X, m*X + b, 'DisplayName', ["s(" t ") = " num2str(m)])
end;

# Instant speeds
for x=2:5;
    y = a*(x.^2);
    m = 2*a*x;
    x, m
end;

# Plot figure
title ("s(t) = 16t^2");
xlabel ("t (seconds)");
ylabel ("s(t)");
grid on;
legend('location', 'west');

uiwait(gcf);