function [xnext] = uni_dyn(x,u,dt)

    v = u(1);
    w = u(2);
    x1 = x(1);
    x2 = x(2);
    theta = x(3);
    
    xnext = [x1+v*cos(theta)*dt; ...
             x2+v*sin(theta)*dt; ...
             theta+w*dt]';
end