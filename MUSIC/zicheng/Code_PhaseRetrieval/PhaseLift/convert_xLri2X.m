function X = convert_xLri2X(xLri,N)

XD = xLri((end-N+1):end);
nRemain = (length(xLri)-N)/2;
real_XT1 = xLri(1:nRemain);
imag_XT1 = xLri((nRemain+1):(2*nRemain));
XT1 = real_XT1 + 1j*imag_XT1;
vecX = zeros(N^2,1);
nd = 0;
for n = 1:(N-1)
    idDiag = n+(n-1)*N;
    vecX(idDiag) = XD(n)/2;
    idLower = ((n+1):N)+(n-1)*N;
    id = nd + (1:length(idLower));
    vecX(idLower) = XT1(id);
    nd = nd + length(idLower);
end
vecX(end) = XD(N);

X1 = reshape(vecX,N,N);
X2 = X1';
X = X1 + X2;