function c=addlogs(a,b)
% Add numbers represented by their logarithms.
% Computes log(exp(a)+exp(b)) in such a fashion that it works even
% when a and b have large magnitude.
if a>b
  c = a + log(1+exp(b-a));
else
  c = b + log(1+exp(a-b));
end

