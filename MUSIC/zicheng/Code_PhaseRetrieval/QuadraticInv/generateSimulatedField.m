function [mat_am,b,xTrue] = generateSimulatedField()

N0 = 12;
lambda = 1e-3;
beta = N0/(1.9*lambda);
vecn = ((-N0):N0).';
xTrue = 10*exp(1j*vecn);
N = 2*N0+1;
% Position of observation points
rho1 = 3*lambda;
rho2 = 5*lambda;
rho3 = 7*lambda;
rho4 = 9*lambda;
vec_rho = [rho1;rho2;rho3;rho4];
M0 = 49;
vec_phi = (1:M0).'*2*pi/M0;
[mat_rho,mat_phi] = meshgrid(vec_rho,vec_phi);
M = length(mat_rho(:));
% Electric field at observation points
rho_obs = repmat(mat_rho(:),1,N);
phi_obs = repmat(mat_phi(:),1,N);
mat_n = repmat(vecn.',M,1);
mat_am = conj(besselh(mat_n,2,beta*rho_obs).*exp(1j*mat_n.*phi_obs));
vecE = conj(mat_am)*xTrue;

b = vecE.*conj(vecE);
b = real(b);