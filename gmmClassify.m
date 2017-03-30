dir_test = 'speechdata/Testing';
dir_train = 'speechdata/Training';

max_iter = 10;
epsilon = 1.0;
M = 8;
D = 14;

test_files = dir([dir_test, filesep, '*mfcc']);

gmms = gmmTrain(dir_train, max_iter, epsilon, M);

N = length(gmms);  % N = 30

for i=1:length(test_files)
    test = textread([dir_test, filesep, test_files(i).name], '%s', 'delimiter', '\n');
    test = strtrim(test);
    test = cellfun(@strsplit, test, 'un', 0);
    test = cellfun(@str2double, test, 'un', 0);
    test = cell2mat(test).';
    
    top5_vals = ones(1, 5)*(-Inf);
    top5_names = {'', '', '', '', ''};
    
    for n=1:N
        gmm = gmms{n};
        L = 0;  % likelihood
        T = length(test);
        
        % calculate likelihood
        for t=1:T
            x = test(:, t);
            total = 0;

            for m=1:M
                omega = gmm.weights(m);
                mu = gmm.means(:, m);
                sigma = diag(gmm.cov(:, :, m));

                denom = (2 * pi)^(D / 2) * prod(sigma)^(0.5);
                b = exp( -0.5 * sum( (x - mu).^2 ./ sigma ) ) / ( denom );

                total = total + omega * b;
            end

            L = L + log(total);
        end
        
        [min_val, min_ind] = min(top5_vals);
        if L > min_val
            top5_vals(min_ind) = L;
            top5_names{min_ind} = gmm.name;
        end
    end
    
    [top5_vals, top5_inds] = sort(top5_vals, 'descend');
    top5_names = top5_names(top5_inds);
    
    file_name = strsplit(test_files(i).name, '.');
    file_name = [file_name{1}, '.lik'];
    fileID = fopen(file_name, 'w');
    
    for j=1:5
        fprintf(fileID, '%s\n', top5_names{j});
    end
    
    fclose(fileID);
end