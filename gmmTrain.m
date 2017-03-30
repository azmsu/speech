function gmms = gmmTrain( dir_train, max_iter, epsilon, M )
% gmmTain
%
%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture

% get subdirectories
parent_dir = dir(dir_train);
sub_dir_names = {parent_dir([parent_dir.isdir]).name};
sub_dir_names = sub_dir_names(~ismember(sub_dir_names,{'.','..'}));

N = length(sub_dir_names);  % N = 30
gmms = cell(1, N);
train = {};

D = 14;

% initialize gmms
for n=1:N
    % get ith sub directory
    sub_dir = dir([dir_train, filesep, sub_dir_names{n}, filesep, '*mfcc']);
    
    train_n = {};
    for i=1:length(sub_dir)
        lines = textread([dir_train, filesep, sub_dir_names{n}, filesep, sub_dir(i).name], '%s', 'delimiter', '\n');
        lines = strtrim(lines);
        lines = cellfun(@strsplit, lines, 'un', 0);
        lines = cellfun(@str2double, lines, 'un', 0);
        train_n = [train_n; lines];
    end
    train.(sub_dir_names{n}) = cell2mat(train_n).';
    
    gmm = struct();
    
    gmm.name = sub_dir_names{n};
    gmm.weights = ones(1, M)/M;
    gmm.means = train.(sub_dir_names{n})(:, randi(length(train.(sub_dir_names{n})), 1, M));
    gmm.cov = repmat(eye(D), 1, 1, M);
    
    gmms{n} = gmm;
end

i = 1;
improvement = Inf;
prev_L = -Inf;
while i <= max_iter && improvement >= epsilon
%     min_L = Inf;
    L = 0;  % likelihood
    
    for n=1:N
        train_n = train.(sub_dir_names{n});
        T = length(train_n);
        gmm = gmms{n};
        probs = zeros(M, T);  % matrix to store probabilities
%         L = 0;  % likelihood
        
        % maximum likelihood
        for t=1:T
            x = train_n(:, t);
            total = 0;

            for m=1:M
                omega = gmm.weights(m);
                mu = gmm.means(:, m);
                sigma = diag(gmm.cov(:, :, m));

                denom = (2 * pi)^(D / 2) * prod(sigma)^(0.5);
                b = exp( -0.5 * sum( (x - mu).^2 ./ sigma ) ) / ( denom );

                probs(m, t) = omega * b;
                total = total + omega * b;
            end

            L = L + log(total);
            probs(:, t) = probs(:, t)./total;
        end
        
%         min_L = min(L, min_L);

        % update parameters
        for m=1:M
            prob_sum = sum(probs(m, :));

            omega = prob_sum/T;
            mu = sum(repmat(probs(m, :), D, 1) .* train_n, 2) ./ prob_sum;
            sigma = diag(sum(repmat(probs(m, :), D, 1) .* train_n.^2, 2) ./ prob_sum - mu.^2);

            gmms{n}.weights(m) = omega;
            gmms{n}.means(:, m) = mu;
            gmms{n}.cov(:, :, m) = sigma;
        end
    end
    
%     improvement = min_L - prev_L;
%     prev_L = min_L;
    improvement = L - prev_L;
    prev_L = L;
    i = i + 1;
end
return
