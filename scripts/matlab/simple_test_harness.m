function result = simple_test_harness(simdata, yaml_file)
% SIMPLE_TEST_HARNESS - Basic validation of test statistics
%
% Takes SimBiology data and YAML template, computes test statistic,
% and checks if it falls within the expected 95% CI.
%
% Usage: result = simple_test_harness(simdata, 'test_stat.yaml')

fprintf('Testing: %s\n', yaml_file);

% 1. Parse YAML to get expected CI and MATLAB code
expected_ci = parse_yaml_ci(yaml_file);
matlab_code = parse_yaml_code(yaml_file);

% 2. Extract species data and run test statistic function
test_value = run_test_statistic(simdata, matlab_code);

% 3. Check if within 95% CI
within_ci = (test_value >= expected_ci(1)) && (test_value <= expected_ci(2));

% 4. Results
result = struct();
result.test_value = test_value;
result.expected_ci = expected_ci;
result.within_ci = within_ci;
result.status = if_else(within_ci, 'PASS', 'FAIL');

fprintf('  Computed: %.4f\n', test_value);
fprintf('  Expected CI: [%.4f, %.4f]\n', expected_ci(1), expected_ci(2));
fprintf('  Result: %s\n', result.status);
end

function ci95 = parse_yaml_ci(yaml_file)
% Extract 95% CI from YAML file
content = fileread(yaml_file);
pattern = 'ci95:\s*\[([0-9.-]+),\s*([0-9.-]+)\]';
tokens = regexp(content, pattern, 'tokens', 'once');
if ~isempty(tokens)
    ci95 = [str2double(tokens{1}), str2double(tokens{2})];
else
    error('Could not find ci95 in YAML file');
end
end

function code = parse_yaml_code(yaml_file)
% Extract MATLAB code from YAML file
content = fileread(yaml_file);
pattern = 'code:\s*\|\s*\n(.*?)(?=\n\S|\n\n|$)';
tokens = regexp(content, pattern, 'tokens', 'once', 'dotexceptnewline');
if ~isempty(tokens)
    code = tokens{1};
else
    error('Could not find MATLAB code in YAML file');
end
end

function test_value = run_test_statistic(simdata, matlab_code)
% Execute the test statistic function from YAML

% Write function to temp file
temp_file = fullfile(tempdir, 'temp_test_function.m');
fid = fopen(temp_file, 'w');
fprintf(fid, '%s\n', matlab_code);
fclose(fid);

% Add temp dir to path
addpath(tempdir);

try
    % Get time vector
    time = simdata.Time;

    % Extract species data (assumes selectbyname function exists)
    % For tumor volume test
    if contains(matlab_code, 'V_T_TumorVolume')
        species = selectbyname(simdata, 'V_T.TumorVolume');
        test_value = temp_compute_test_statistic(time, species.Data);

    % For CD8/Treg ratio test
    elseif contains(matlab_code, 'V_T_T_eff')
        cd8_species = selectbyname(simdata, 'V_T.T_eff');
        treg_species = selectbyname(simdata, 'V_T.T_reg');
        test_value = temp_compute_test_statistic(time, cd8_species.Data, treg_species.Data);

    else
        error('Unknown test statistic type in MATLAB code');
    end

catch ME
    warning('Error running test statistic: %s', ME.message);
    test_value = NaN;
end

% Cleanup
delete(temp_file);
rmpath(tempdir);
end

function result = if_else(condition, true_val, false_val)
% Simple conditional helper
if condition
    result = true_val;
else
    result = false_val;
end
end