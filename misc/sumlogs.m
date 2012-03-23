function y = sumlogs(x)
% Sum of vector where numbers are represented by their logarithms.
% Computes log(sum(exp(a))) in such a fashion that it works even
% when elements have large magnitude.
y=x(1);
for k=2:length(x)
  y=addlogs(y,x(k));
end
