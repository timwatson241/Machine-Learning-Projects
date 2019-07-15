function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);
inside = 0;

for i = 1:length(x1)
    inside = inside + (x1(i)-x2(i)).^2;
end

sim = exp(-inside/(2*sigma^2));

end
