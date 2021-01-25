% This code is realizing Phaselift algorithm described in literature
% Moretta & Pierri, IEEE TAP, Dec. 2019

clear all;
close all;

[mat_am,b,xTrue] = generateSimulatedField;
[ALri,d] = compoMatrix4Inv(mat_am,xTrue);
xEst = algPhaseLift(ALri,d,b);
% xEst = recSol_xEst(:,end);
% vecEst = conj(mat_am)*xEst;

normalized_xTrue = xTrue*conj(xTrue(1))/abs(conj(xTrue(1)));
normalized_xEst = xEst*conj(xEst(1))/abs(conj(xEst(1)));
N0 = (length(xEst)-1)/2;
vecn = (-N0):N0;
figure
hold on;
plot(vecn,angle(normalized_xTrue))
plot(vecn,angle(normalized_xEst),'*')
xlim([-12 12])

figure
hold on;
plot(vecn,abs(normalized_xTrue))
plot(vecn,abs(normalized_xEst),'*')
xlim([-12 12])