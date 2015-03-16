X = randn(100, 5);
results = zeros(2*nf, size(X,2));
for colnum=1:size(X,2)
    col = X(:, colnum);
    res = { ...
      CO_AddNoise(col), ...
      CO_AddNoise(col) ...
     }; 
    % horzcat
    % struct2array
    % nf = numel(fieldnames(CO_AddNoise(randn(100,1))));
    size(res)
    size(results)
end
