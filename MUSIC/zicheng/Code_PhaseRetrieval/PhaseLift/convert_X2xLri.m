function xLri = convert_X2xLri(X)

N = size(X,1);
vecX = X(:);
XT1 = [];
for n = 2:N
    idCol = (n:N)+(n-2)*N;
    XT1 = [XT1;vecX(idCol)];
end

XD = [];
for n = 1:N
    idCol = n+(n-1)*N;
    XD = [XD;vecX(idCol)];
end

xLri = [real(XT1);imag(XT1);real(XD)];