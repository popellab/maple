function [mc_draws, target_mean, target_variance, target_ci95] = generate_calibration_target_from_yaml(yaml_file)
    % Generate calibration target distribution using R bootstrap code from YAML
    %
    % Inputs:
    %   yaml_file - path to YAML file containing derivation_code_r section
    %
    % Outputs:
    %   mc_draws        - bootstrap samples vector
    %   target_mean     - mean of bootstrap distribution
    %   target_variance - variance of bootstrap distribution
    %   target_ci95     - 95% confidence interval [lower, upper]
    %
    % Example:
    %   [samples, mean_val, var_val, ci] = generate_calibration_target_from_yaml('test_stat.yaml')

    % Read YAML file
    fid = fopen(yaml_file, 'r');
    if fid == -1
        error('Could not open YAML file: %s', yaml_file);
    end
    yaml_content = fread(fid, '*char')';
    fclose(fid);

    % Extract R code between ```r and ```
    r_start = strfind(yaml_content, '```r');
    r_end = strfind(yaml_content, '```');

    if isempty(r_start) || length(r_end) < 2
        error('Could not find R code block in YAML file');
    end

    r_code_start = r_start(1) + 4; % Skip '```r'
    r_code_end = r_end(2) - 1; % Before closing ```
    r_code = yaml_content(r_code_start:r_code_end);

    % Clean up the R code
    r_code = strtrim(r_code);

    % Write R code to temporary file
    temp_r_file = tempname;
    temp_r_file = [temp_r_file '.R'];

    fid = fopen(temp_r_file, 'w');
    fprintf(fid, '%s\n', r_code);

    % Add code to fix BCa CI issue and save results to CSV for MATLAB to read
    fprintf(fid, '\n# Fix BCa CI issue - use percentile method if BCa fails\n');
    fprintf(fid, 'tryCatch({\n');
    fprintf(fid, '  bca_ci <- boot.ci(boot_out, type = "bca", conf = 0.95)\n');
    fprintf(fid, '  ci95_stat <- c(bca_ci$bca[4], bca_ci$bca[5])  # [lower, upper]\n');
    fprintf(fid, '}, error = function(e) {\n');
    fprintf(fid, '  cat("BCa CI failed, using percentile method\\n")\n');
    fprintf(fid, '  perc_ci <- boot.ci(boot_out, type = "perc", conf = 0.95)\n');
    fprintf(fid, '  ci95_stat <<- c(perc_ci$percent[4], perc_ci$percent[5])  # [lower, upper]\n');
    fprintf(fid, '})\n');
    fprintf(fid, '\n# Save results for MATLAB\n');
    fprintf(fid, 'write.csv(data.frame(mc_draws=mc_draws), "mc_draws.csv", row.names=FALSE)\n');
    fprintf(fid, 'write.csv(data.frame(mean=mean_stat, variance=variance_stat, ci_lower=ci95_stat[1], ci_upper=ci95_stat[2]), "stats.csv", row.names=FALSE)\n');
    fclose(fid);

    % Execute R script
    fprintf('Executing R bootstrap code...\n');
    [status, result] = system(['R --vanilla < ' temp_r_file]);

    if status ~= 0
        delete(temp_r_file);
        error('R execution failed with error:\n%s', result);
    end

    % Read results back into MATLAB
    if ~exist('mc_draws.csv', 'file') || ~exist('stats.csv', 'file')
        delete(temp_r_file);
        error('R execution did not produce expected output files');
    end

    mc_data = readtable('mc_draws.csv');
    stats_data = readtable('stats.csv');

    mc_draws = mc_data.mc_draws;
    target_mean = stats_data.mean;
    target_variance = stats_data.variance;
    target_ci95 = [stats_data.ci_lower, stats_data.ci_upper];

    fprintf('Bootstrap completed: %d samples generated\n', length(mc_draws));

    % Clean up temporary files
    delete('mc_draws.csv');
    delete('stats.csv');
    delete(temp_r_file);
end