function xEst = algQuadraticInv(mat_am,b)

% iterative retrieval
maxIte = 1e4;
N = size(mat_am,2);
randVal = rand([N,2])*60-30;
i_x = randVal(:,1) + 1j*randVal(:,2);
thre_absErr = 1e-3;
thre_relErr = 1e-5;
thre_relErr_xEst = 1e-5;

vec_absErr = zeros(maxIte,1);
vec_relErr = ones(maxIte,1);
vec_relErr_xEst = ones(maxIte,1);
recSol_xEst = zeros(N,maxIte);
for iter = 1:maxIte
    g = compGradient(i_x,mat_am,b);
    t = compuStepSize(i_x,g,mat_am,b);
    i_x = i_x - t*g;
    
    recSol_xEst(:,iter) = i_x;
    ax = conj(conj(mat_am)*i_x);
    vec_absErr(iter) = norm(b-(abs(ax).^2))^2;
    

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

xEst = i_x;


figure
semilogy(1:length(vec_absErr),vec_absErr)
figure
semilogy(1:length(vec_relErr),vec_relErr)
figure
semilogy(1:length(vec_relErr_xEst),vec_relErr_xEst)