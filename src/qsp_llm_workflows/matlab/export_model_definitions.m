function export_model_definitions(model_file, temp_dir, model_type)
% Export model definitions for the metadata workflow
%
% Inputs:
%   model_file - path to the model file (MATLAB script .m or SimBiology project .sbproj)
%   temp_dir - directory to save the exported CSV files
%   model_type - (optional) 'matlab_script' or 'simbiology_project'.
%                If not provided, inferred from file extension.
%
% Outputs:
%   Creates three CSV files:
%   - simbio_parameters.csv - parameter definitions table
%   - simbio_species.csv - species definitions table
%   - model_context.csv - parameter-reaction context table

    try
        % Infer model type from extension if not provided
        if nargin < 3 || isempty(model_type)
            [~, ~, ext] = fileparts(model_file);
            if strcmp(ext, '.sbproj')
                model_type = 'simbiology_project';
            else
                model_type = 'matlab_script';
            end
        end

        % Load the model based on type
        fprintf('Loading model from: %s (type: %s)\n', model_file, model_type);

        if strcmp(model_type, 'simbiology_project')
            % Load from SimBiology project file
            fprintf('Loading SimBiology project...\n');
            proj = sbioloadproject(model_file);

            % sbioloadproject returns a struct with model fields
            % Get field names and extract first model
            modelFields = fieldnames(proj);

            if isempty(modelFields)
                error('SimBiology project contains no models');
            end

            % Access first model using dynamic field access
            model = proj.(modelFields{1});
            fprintf('Loaded model: %s\n', model.Name);

            if length(modelFields) > 1
                warning('Project contains %d models. Using first model: %s', ...
                    length(modelFields), model.Name);
            end
        else
            % Load from MATLAB script (default behavior)
            fprintf('Running MATLAB model script...\n');
            run(model_file);

            if ~exist('model', 'var')
                error('Model script did not create a variable named "model"');
            end
        end

        % Export parameters
        fprintf('Exporting parameters...\n');
        parameters = sbioselect(model, 'Type', 'parameter');

        if isempty(parameters)
            warning('No parameters found in the model');
            parameterNames = {};
            parameterValues = {};
            parameterUnits = {};
            parameterNotes = {};
            parameterDerivedFrom = {};
        else
            % Get all repeated assignment rules to identify parameters to skip
            fprintf('Identifying parameters updated via repeatedAssignment rules...\n');
            repeatedAssignmentTargets = {};

            rules = sbioselect(model, 'Type', 'rule');
            if ~isempty(rules)
                for i = 1:length(rules)
                    if strcmp(rules(i).RuleType, 'repeatedAssignment')
                        % Extract target parameter name from the rule
                        ruleExpr = rules(i).Rule;
                        % Find the left side of the assignment (before '=')
                        equalPos = strfind(ruleExpr, '=');
                        if ~isempty(equalPos)
                            targetParam = strtrim(ruleExpr(1:equalPos(1)-1));
                            % Remove any compartment prefix (e.g., 'V_T.param' -> 'param')
                            dotPos = strfind(targetParam, '.');
                            if ~isempty(dotPos)
                                targetParam = targetParam(dotPos(end)+1:end);
                            end
                            repeatedAssignmentTargets{end+1} = targetParam;
                        end
                    end
                end
            end

            fprintf('Found %d parameters updated via repeatedAssignment: %s\n', ...
                length(repeatedAssignmentTargets), strjoin(repeatedAssignmentTargets, ', '));

            % Export ALL parameters (including repeatedAssignment targets) but mark them
            parameterNames = {parameters.Name};
            parameterValues = {parameters.Value};
            parameterUnits = {parameters.Units};
            parameterNotes = {parameters.Notes};

            % Extract derivedFrom from UserData
            parameterDerivedFrom = strings(1, length(parameters));
            for i = 1:length(parameters)
                userData = parameters(i).UserData;
                if iscell(userData) && ~isempty(userData)
                    % Join cell array to comma-separated string
                    derivedStr = strjoin(userData, ',');
                    parameterDerivedFrom(i) = string(derivedStr);
                elseif ~isempty(userData)
                    % Convert to single string
                    if ischar(userData)
                        parameterDerivedFrom(i) = string(userData);
                    elseif isstring(userData)
                        % Join multiple strings if it's a string array
                        parameterDerivedFrom(i) = strjoin(userData, ',');
                    else
                        % Unknown type, convert to string
                        parameterDerivedFrom(i) = string(parameterNames{i});
                    end
                else
                    % Default to parameter name if UserData not set
                    parameterDerivedFrom(i) = string(parameterNames{i});
                end
            end

            % Mark repeatedAssignment targets
            isRepeatedAssignment = false(size(parameterNames));
            for i = 1:length(parameterNames)
                if ismember(parameterNames{i}, repeatedAssignmentTargets)
                    isRepeatedAssignment(i) = true;
                    fprintf('  Marking parameter as repeatedAssignment: %s\n', parameterNames{i});
                end
            end

            fprintf('Exported %d parameters (%d are repeatedAssignment targets)\n', ...
                length(parameterNames), sum(isRepeatedAssignment));
        end

        T_params = table(parameterNames', parameterValues', parameterUnits', parameterNotes', parameterDerivedFrom', isRepeatedAssignment', ...
            'VariableNames', {'Name', 'Value', 'Units', 'Notes', 'DerivedFrom', 'IsRepeatedAssignment'});

        param_file = fullfile(temp_dir, 'simbio_parameters.csv');
        writetable(T_params, param_file, 'QuoteStrings', true);
        fprintf('Parameters exported to: %s\n', param_file);

        % Export species
        fprintf('Exporting species...\n');
        species = sbioselect(model, 'Type', 'species');

        if isempty(species)
            warning('No species found in the model');
            speciesNames = {};
            speciesUnits = {};
            speciesNotes = {};
            speciesCompartments = {};
        else
            speciesNames = {species.Name};
            speciesUnits = {species.Units};
            speciesNotes = {species.Notes};
            % Get compartment names (Parent is the compartment object)
            speciesCompartments = cell(1, length(species));  % Row vector like the others
            for i = 1:length(species)
                if ~isempty(species(i).Parent)
                    speciesCompartments{i} = species(i).Parent.Name;
                else
                    speciesCompartments{i} = 'unknown';
                end
            end
        end

        % Create table (handle empty case)
        if isempty(speciesNames)
            T_species = table(cell(0,1), cell(0,1), cell(0,1), cell(0,1), ...
                'VariableNames', {'Name', 'Units', 'Notes', 'Compartment'});
        else
            T_species = table(speciesNames', speciesUnits', speciesNotes', speciesCompartments', ...
                'VariableNames', {'Name', 'Units', 'Notes', 'Compartment'});
        end

        species_file = fullfile(temp_dir, 'simbio_species.csv');
        writetable(T_species, species_file);
        fprintf('Species exported to: %s\n', species_file);

        % Export model context using parameterReactionTableExtended
        fprintf('Exporting model context...\n');
        T_context = parameterReactionTableExtended(model);

        context_file = fullfile(temp_dir, 'model_context.csv');
        writetable(T_context, context_file);
        fprintf('Model context exported to: %s\n', context_file);

        fprintf('Model export completed successfully\n');

    catch ME
        fprintf('Error in model export: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        rethrow(ME);
    end
end
