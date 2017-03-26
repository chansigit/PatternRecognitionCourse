function components = em_gmm(data, k)
% INPUT
% data: num-by-dim matrix, num is the number of data points
% dim is the dimension of a data point
% k: the number of Gaussian components
% OUTPUT
% components: a struct array with fields 'mu', 'sigma', 'p'
% 'mu' is the mean of a component
% 'sigma' is the covariance of a component
% 'p' is the mixing proportions of each component
[num, dim] = size(data);
labels = kmeans_clustering(data, k); % initialize with kmeans
p = ones(1, k) / k;
mu = zeros(k, dim);
sigma = zeros(dim, dim, k);
for i = 1:k
    mu(i, :) = mean(data(labels == i, :), 1);
    sigma(:, :, i) = eye(dim);
end
pdf = zeros(num, k);
d = inf; gamma0 = zeros(num, k); eps = 1e-5;
while d > eps
    % the E-step
    for i = 1:k
        pdf(:, i) = gaussian_pdf(data, mu(i, :),sigma(:, :, i));
    end
    gamma = bsxfun(@times, p, pdf);
    gamma = bsxfun(@times, gamma, 1./sum(gamma, 2));
    d = sum((gamma(:) - gamma0(:)) .^ 2);
    gamma0 = gamma;
    % the M-step
    n = sum(gamma, 1);
    for i = 1:k
        mu(i, :) = sum(bsxfun(@times, gamma(:, i), data), 1)/ n(i);
        tem = bsxfun(@minus, data, mu(i, :));
        sigma(:, :, i) = tem' * bsxfun(@times, tem,gamma(:, i)) / n(i);
    end
    p = n / num;
end
components.mu = mu;
components.sigma = sigma;
components.p = p;
end

