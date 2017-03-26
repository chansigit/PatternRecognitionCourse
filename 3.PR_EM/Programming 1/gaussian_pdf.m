function p = gaussian_pdf(x, mu, sigma)
% x: num-by-dim
% mu: 1-by-dim
% sigma: dim-by-dim
[~, dim] = size(x);
d = bsxfun(@minus, x, mu);
p = exp(-sum(d .* (d / sigma'), 2) / 2) / sqrt(det(sigma) * (2 * pi) ^ dim);
end