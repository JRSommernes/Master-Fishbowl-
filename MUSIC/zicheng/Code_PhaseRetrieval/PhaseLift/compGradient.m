function g = compGradient(xLri,ALri,gamma,d,b)

g = 2*(ALri.')*ALri*xLri-2*(ALri.')*b+gamma*d;