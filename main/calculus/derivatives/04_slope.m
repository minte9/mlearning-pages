# Plot the slope gradient
#
# Function:     f(b) = ax^2
# Derivative:   f'(x) = 2ax 
# Gradient:     g(x) = mx + b
# Coeficient:   m (or slope of the line)
# Intercept:    b (where the line crosses y-axis)

# Falliing object s(t) = 16t^2
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

    m = 2*a*x;      # slope coeficient (derivative)
    b = y - m*x;    # intercept in y = mx + b
    X = x:x+2;
    t = num2str(x);
    plot(X, m*X + b, 'DisplayName', ["s(" t ") = " num2str(m)])
end;

# Instant speeds
for x=2:5;
    y = a*(x.^2);   # 16t^2             = 64, 144, 256, 400
    m = 2*a*x;      # 32t               = 64, 96, 128, 160
    b = y - m*x;    # 16t^2 - (32t)t    = -64, -144, -256, -400    
    x, y, m, b
    disp('')
end;

# Plot figure
title ("s(t) = 16t^2");
xlabel ("t (seconds)");
ylabel ("s(t)");
grid on;
legend('location', 'west');

uiwait(gcf);