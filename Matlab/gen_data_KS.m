%% Kuramoto-Sivashinsky equation and chaos
nn = 511;
steps = 250;

dom = [0 32*pi]; x = chebfun('x',dom); tspan = linspace(0,100,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) - diff(u,2) - diff(u,4);
S.nonlin = @(u) - 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = cos(x/16).*(1+sin(x/16));
% S.init = -sin(pi*x/50);
u = spin(S,nn,1e-4);

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(0,32*pi,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
save('ks.mat','t','x','usol')
