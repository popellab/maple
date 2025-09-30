function test_stat_value = compute_test_statistic_from_yaml(yaml_file, time, simdata_output)
    % Compute test statistic using MATLAB code from YAML file
    %
    % Inputs:
    %   yaml_file       - path to YAML file containing test statistic definition
    %   time           - time vector from simulation
    %   simdata_output - simulation output vector (e.g., V_T_TumorVolume)
    %
    % Output:
    %   test_stat_value - computed test statistic value
    %
    % Example:
    %   test_stat = compute_test_statistic_from_yaml('test_stat.yaml', time, V_T_TumorVolume)

    % Read YAML file
    fid = fopen(yaml_file, 'r');
    if fid == -1
        error('Could not open YAML file: %s', yaml_file);
    end
    yaml_content = fread(fid, '*char')';
    fclose(fid);

    % Find the model_output code section
    code_start = strfind(yaml_content, 'code: |');
    if isempty(code_start)
        error('Could not find model_output code section in YAML');
    end

    % Extract code section until next YAML section
    remaining_content = yaml_content(code_start:end);
    lines = strsplit(remaining_content, '\n');
    end_idx = length(lines);

    for i = 2:length(lines) % Start from line 2 (skip 'code: |' line)
        line = lines{i};
        % If line starts with non-whitespace (new YAML section) or is '# '
        if ~isempty(line) && (line(1) ~= ' ' || startsWith(strtrim(line), '#'))
            end_idx = i - 1;
            break;
        end
    end

    code_section = strjoin(lines(1:end_idx), '\n');

    % Extract and clean MATLAB code
    code_lines = strsplit(code_section, '\n');
    matlab_lines = {};

    for i = 2:length(code_lines) % Skip 'code: |' line
        line = code_lines{i};
        % Remove YAML indentation (up to 6 spaces)
        cleaned_line = regexprep(line, '^\s{0,6}', '');
        if ~isempty(strtrim(cleaned_line))
            matlab_lines{end+1} = cleaned_line;
        end
    end

    matlab_code = strjoin(matlab_lines, '\n');

    % Write MATLAB code to temporary file and add execution wrapper
    temp_matlab_file = 'temp_test_stat.m';
    fid = fopen(temp_matlab_file, 'w');
    fprintf(fid, '%s\n\n', matlab_code);

    % Add execution code that calls the function and saves result
    fprintf(fid, '%% Execution wrapper\n');
    fprintf(fid, 'load(''simdata.mat'');\n');
    fprintf(fid, 'computed_test_stat = compute_test_statistic(time, simdata_output);\n');
    fprintf(fid, 'save(''test_stat_result.mat'', ''computed_test_stat'');\n');
    fclose(fid);

    % Save simulation data for MATLAB script to use
    save('simdata.mat', 'time', 'simdata_output');

    try
        % Execute MATLAB script directly in current session
        clear compute_test_statistic;
        run('temp_test_stat.m');

        if exist('test_stat_result.mat', 'file')
            load('test_stat_result.mat');
            test_stat_value = computed_test_stat;

            % Clean up temp files
            delete('test_stat_result.mat');
            delete('simdata.mat');
        else
            error('MATLAB execution did not produce output file');
        end
    catch ME
        % Clean up on error
        if exist('test_stat_result.mat', 'file'), delete('test_stat_result.mat'); end
        if exist('simdata.mat', 'file'), delete('simdata.mat'); end
        error('Failed to execute MATLAB code: %s', ME.message);
    end

    % Clean up script
    delete(temp_matlab_file);
end