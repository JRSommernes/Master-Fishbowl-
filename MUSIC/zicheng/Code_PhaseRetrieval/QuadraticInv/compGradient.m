function g = compGradient(x,mat_am,b)

vecE = conj(mat_am)*x;
b_est = vecE.*conj(vecE);
g = (mat_am.')*diag(b_est-b)*conj(mat_am)*x;
g = g/size(mat_am,1);