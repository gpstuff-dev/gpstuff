function [ea] = ext_auc(P,tt,t)
% [ea] = EXT_AUC(P,tt,t)
%  Given probability matrix P os size n x size(tt,2), tt time vector of predictions and t time
%  where we want to evaluate our EXT_AUC(t) returns EXT_AUC proposed by
%  L. E. Chambless, C. P. Cummiskey and G.Cui in Satist. Med. 2011, 30
%  22-38
% 
% ip=inputParser;
% ip.addRequired('P',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
% ip.addRequired('tt', @(x) isreal(x) && all(isfinite(x(:))))
% ip.addRequired('t', @(x) isreal(x) && all(isfinite(x(:))))
% ip.parse(P,tt,t);

    [n,nin]=size(P);
    S=1-P;
    D=-diff([; ones(n,1) S],1,2);
    sd=tt(2)-tt(1);
    
    if isempty(find(tt==t,1))
        error('Could not find your time value in the model prediction matrix')
    else
        I=find(tt==t);
    end
    
    if size(tt,2) > size(P,2)
        error('prediction matrix P must have number of columns equal to length(tt)')
    end
    
    comp=bsxfun(@gt,P(:,I),P(:,I)');
    %inner=D*S';
    den=(2/(1-mean(S(:,I)).^2))*(1/n^2);
    %ea=den*sum(sum(inner.*comp));
    extauc=0;
    for i=1:n
        for j=1:n
            if comp(i,j) == 1
                for d=1:size(tt,2)
                    extauc=extauc + sum(D(i,d).*S(j,d)) + (sd/2)*(D(i,d));
                end
            end
        end
    end
    
    ea=extauc*den;

end

