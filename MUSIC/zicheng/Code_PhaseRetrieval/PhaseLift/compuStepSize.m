function t = compuStepSize(xLri,g,ALri,gamma,d,b)

y = ALri*xLri;

t = (2*((y-b).')*ALri*g+gamma*(d.')*g)/...
    (2*(g.')*(ALri.')*ALri*g);