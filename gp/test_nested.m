function f = test_nested(F)
    
    x = [];
    y = [];
    f = @inner;
    function out = inner(in)
        ind = find(in == x);
        if isempty(ind)
            out = sin(in);
            x(end+1) = in;
            y(end+1) = out;
            fprintf('evaluate')
        else
            out = y(ind);
            fprintf('memorize')
        end
    end
end
