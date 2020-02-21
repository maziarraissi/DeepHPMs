%% Kuramoto-Sivashinsky equation and chaos
nn = 512;
steps = 500;

dom = [-5 5]; x = chebfun('x',dom); tspan = linspace(0,pi/2,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) 0.5*1i*diff(u,2);
S.nonlin = @(u) 1i*abs(u).^2.*u; % spin cannot parse "u.*diff(u)"
S.init = 2*sech(x);
u = spin(S,nn,pi/2*1e-6);

usol = zeros(nn,steps+1);
for it = 1:steps+1
    usol(:,it) = u{it}.values;
end

x = linspace(-5,5,nn+1);
x = x(1:end-1);
t = tspan;
pcolor(t,x,abs(usol)); shading interp, axis tight, colormap(jet);
save('NLS.mat','t','x','usol')