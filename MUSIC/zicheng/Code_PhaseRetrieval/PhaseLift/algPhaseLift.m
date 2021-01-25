function xEst = algPhaseLift(ALri,d,b)

N = sqrt(size(ALri,2));

% iterative retrieval
maxIte = 1e4;
gamma = 0.5;
i_xLri = rand([size(ALri,2),1])*40-20;
% save('i_xLri','i_xLri')
% load i_xLri.mat;
thre_absErr = 1e-3;
thre_relErr = 1e-5;
thre_relErr_xEst = 1e-5;

vec_absErr = zeros(maxIte,1);
vec_relErr = ones(maxIte,1);
vec_relErr_xEst = ones(maxIte,1);
recSol_xEst = zeros(size(ALri,2),maxIte);
for iter = 1:maxIte
    g = compGradient(i_xLri,ALri,gamma,d,b);
    t = compuStepSize(i_xLri,g,ALri,gamma,d,b);
    i_xLri = i_xLri - t*g;
    
%     X = convert_xLri2X(i_xLri,N);
%     [V,D] = eig(X);
%     idNegEigVal = find(diag(D)>=0);
%     X = V(:,idNegEigVal)*D(idNegEigVal,idNegEigVal)*(V(:,idNegEigVal)');
%     i_xLri = convert_X2xLri(X);
    
%     recSol_xEst(:,iter) = sqrt(D(end,end))*V(:,end);
    recSol_xEst(:,iter) = i_xLri;
    vec_absErr(iter) = norm(b-ALri*i_xLri)^2;
    

    if iter > 1
        vec_relErr(iter) = abs(vec_absErr(iter)-vec_absErr(iter-1))/vec_absErr(iter);
        vec_relErr_xEst(iter) = max(abs(recSol_xEst(:,iter)-recSol_xEst(:,iter-1))./abs(recSol_xEst(:,iter)));
    end
    
    if (vec_absErr(iter) < thre_absErr) || (vec_relErr(iter) < thre_relErr) || (vec_relErr_xEst(iter) < thre_relErr_xEst)
        vec_absErr((iter+1):end) = [];
        vec_relErr((iter+1):end) = [];
        vec_relErr_xEst((iter+1):end) = [];
        recSol_xEst(:,(iter+1):end) = [];
        break;
    end
end

figure
semilogy(1:length(vec_absErr),vec_absErr)
figure
semilogy(1:length(vec_relErr),vec_relErr)
figure
semilogy(1:length(vec_relErr_xEst),vec_relErr_xEst)

X = convert_xLri2X(i_xLri,N);
[V,D] = eig(X);
xEst = sqrt(D(end,end))*V(:,end);
