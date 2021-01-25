function [ALri,d] = compoMatrix4Inv(mat_am,xTrue)

[M,N] = size(mat_am);
% compose matrix AL
AL = nan(M,N^2);
for m = 1:M
    Am = (mat_am(m,:).') * conj(mat_am(m,:));
    AmT = Am.';
    AL(m,:) = AmT(:).';
end
% norm(b-AL*XTrue(:))

XTrue = xTrue*xTrue';
% compose matix F
F = [];
XT1 = [];
vecXTrue = XTrue(:);
for n = 2:N
    idCol = (n:N)+(n-2)*N;
    F = [F,AL(:,idCol)];
    XT1 = [XT1;vecXTrue(idCol)];
end

% compose matrix H
H = [];
XD = [];
for n = 1:N
    idCol = n+(n-1)*N;
    H = [H,AL(:,idCol)];
    XD = [XD;vecXTrue(idCol)];
end
% norm(b-F*XT1-conj(F*XT1)-H*XD)

% compose matrix ALri
ALri = [2*real(F),-2*imag(F),real(H)];
xLri = [real(XT1);imag(XT1);real(XD)];
% norm(b-ALri*xLri)

d = [zeros(1,size(F,2)*2),ones(1,N)].';