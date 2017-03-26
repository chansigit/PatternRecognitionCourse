function labels = kmeans_clustering(data, k)
% INPUT
% data: num-by-dim matrix, num is the number of data points, dim
% is the dimension of a point.
% k: the number of clusters
% OUTPUT
% labels: the corresponding label to each data point
[num, ~] = size(data);
% randomly select k data points as initial points
% only need one line in matlab 2013 and could be more
% efficient:
% ind = randsample(num, k);
ind = randperm(num);
ind = ind(1:k);
centers = data(ind, :);
d = inf;
labels = nan(num, 1);
while d > 0
    labels0 = labels;
    dist = pdist2(data, centers);
    [~, labels] = min(dist, [], 2);
    d = sum(labels0 ~= labels);
    for i = 1:k
        centers(i, :) = mean(data(labels == i, :), 1);
    end
end
end
