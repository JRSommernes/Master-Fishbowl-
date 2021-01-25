function t = compuStepSize(x,g,mat_am,b)

% compute coefficients of t
adx = conj(mat_am)*g;
c3 = sum(abs(adx).^4);

ax = conj(conj(mat_am)*x);
Re_ax = real(ax.*adx);
c2 = -3*sum((abs(adx).^2).*Re_ax);

c1 = sum(2*Re_ax.^2 + (abs(adx).^2).*((abs(ax).^2)-b));

c0 = -sum(Re_ax.*((abs(ax).^2)-b));

solt = roots([c3,c2,c1,c0]);
idRealVal = find(abs(imag(solt))<1e-20);
if length(idRealVal) == 3
    dataFitErr = zeros(3,1);
    for it = 1:3
        xnew = x - solt(it)*g;
        ax = conj(conj(mat_am)*xnew);
        dataFitErr(it) = norm((abs(ax).^2)-b);
    end
    [~,idMin] = min(dataFitErr);
    t = solt(idMin);
elseif length(idRealVal) == 1
    t = solt(idRealVal);
end