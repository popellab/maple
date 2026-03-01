function export_model_structure(model_file, output_file, model_type)
% Export complete model structure as JSON for Python ModelStructure
%
% Inputs:
%   model_file  - path to the model file (.m script or .sbproj)
%   output_file - path to output JSON file
%   model_type  - (optional) 'matlab_script' or 'simbiology_project'
%
% Output JSON structure matches Python ModelStructure:
%   {
%     "species": [{name, compartment, base_name, units, description}, ...],
%     "compartments": [{name, volume, volume_units, description}, ...],
%     "parameters": [{name, value, units, description}, ...],
%     "reactions": [{name, reactants, products, rate_law, parameters}, ...]
%   }

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

        % Load the model
        fprintf('Loading model from: %s (type: %s)\n', model_file, model_type);
        model = loadModel(model_file, model_type);
        fprintf('Loaded model: %s\n', model.Name);

        % Build structure
        modelStruct = struct();
        modelStruct.species = exportSpecies(model);
        modelStruct.compartments = exportCompartments(model);
        modelStruct.parameters = exportParameters(model);
        modelStruct.reactions = exportReactions(model);

        % Write JSON
        jsonText = jsonencode(modelStruct, 'PrettyPrint', true);
        fid = fopen(output_file, 'w');
        fprintf(fid, '%s', jsonText);
        fclose(fid);

        fprintf('Exported model structure to: %s\n', output_file);
        fprintf('  Species: %d\n', length(modelStruct.species));
        fprintf('  Compartments: %d\n', length(modelStruct.compartments));
        fprintf('  Parameters: %d\n', length(modelStruct.parameters));
        fprintf('  Reactions: %d\n', length(modelStruct.reactions));

    catch ME
        fprintf('Error in model export: %s\n', ME.message);
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        rethrow(ME);
    end
end

% =========================================================================
% Model loading
% =========================================================================

function model = loadModel(model_file, model_type)
    if strcmp(model_type, 'simbiology_project')
        proj = sbioloadproject(model_file);
        modelFields = fieldnames(proj);
        if isempty(modelFields)
            error('SimBiology project contains no models');
        end
        model = proj.(modelFields{1});
    else
        run(model_file);
        if ~exist('model', 'var')
            error('Model script did not create a variable named "model"');
        end
    end
end

% =========================================================================
% Species export
% =========================================================================

function speciesList = exportSpecies(model)
    species = sbioselect(model, 'Type', 'species');
    speciesList = [];

    if isempty(species)
        return;
    end

    for i = 1:length(species)
        s = species(i);
        compartment = '';
        if ~isempty(s.Parent)
            compartment = char(s.Parent.Name);
        end

        entry = struct();
        entry.name = [compartment '.' char(s.Name)];
        entry.compartment = compartment;
        entry.base_name = char(s.Name);
        entry.units = safeString(s.Units, 'dimensionless');
        entry.description = safeString(s.Notes, '');

        speciesList = [speciesList; entry]; %#ok<AGROW>
    end
end

% =========================================================================
% Compartment export
% =========================================================================

function compartmentList = exportCompartments(model)
    compartments = sbioselect(model, 'Type', 'compartment');
    compartmentList = [];

    if isempty(compartments)
        return;
    end

    for i = 1:length(compartments)
        c = compartments(i);

        entry = struct();
        entry.name = char(c.Name);
        entry.volume = c.Capacity;
        entry.volume_units = safeString(c.CapacityUnits, 'milliliter');
        entry.description = safeString(c.Notes, '');

        compartmentList = [compartmentList; entry]; %#ok<AGROW>
    end
end

% =========================================================================
% Parameter export
% =========================================================================

function parameterList = exportParameters(model)
    parameters = sbioselect(model, 'Type', 'parameter');
    parameterList = [];

    if isempty(parameters)
        return;
    end

    for i = 1:length(parameters)
        p = parameters(i);

        entry = struct();
        entry.name = char(p.Name);
        entry.value = p.Value;
        entry.units = safeString(p.Units, 'dimensionless');
        entry.description = safeString(p.Notes, '');

        parameterList = [parameterList; entry]; %#ok<AGROW>
    end
end

% =========================================================================
% Reaction export
% =========================================================================

function reactionList = exportReactions(model)
    reactions = sbioselect(model, 'Type', 'reaction');
    reactionList = [];

    if isempty(reactions)
        return;
    end

    % Build parameter name set for extraction from rate laws
    params = sbioselect(model, 'Type', 'parameter');
    paramNames = {};
    if ~isempty(params)
        paramNames = {params.Name};
    end

    for i = 1:length(reactions)
        r = reactions(i);

        entry = struct();
        entry.name = char(r.Name);
        if isempty(entry.name)
            % Use reaction equation as name if no name set
            entry.name = char(r.Reaction);
        end

        % Extract reactants as qualified names
        entry.reactants = getQualifiedSpeciesNames(r.Reactants);

        % Extract products as qualified names
        entry.products = getQualifiedSpeciesNames(r.Products);

        % Rate law
        entry.rate_law = '';
        try
            entry.rate_law = char(r.ReactionRate);
        catch
        end

        % Parameters in rate law
        entry.parameters = extractParametersFromRate(entry.rate_law, paramNames);

        reactionList = [reactionList; entry]; %#ok<AGROW>
    end
end

function names = getQualifiedSpeciesNames(speciesArray)
    names = {};
    if isempty(speciesArray)
        return;
    end
    for i = 1:length(speciesArray)
        s = speciesArray(i);
        comp = '';
        if ~isempty(s.Parent)
            comp = char(s.Parent.Name);
        end
        names{end+1} = [comp '.' char(s.Name)]; %#ok<AGROW>
    end
end

function params = extractParametersFromRate(rateLaw, allParamNames)
    % Find which parameters appear in the rate law
    params = {};
    if isempty(rateLaw) || isempty(allParamNames)
        return;
    end

    for i = 1:length(allParamNames)
        pname = allParamNames{i};
        % Match parameter with word boundaries
        pattern = ['(?<![A-Za-z0-9_])' pname '(?![A-Za-z0-9_])'];
        if ~isempty(regexp(rateLaw, pattern, 'once'))
            params{end+1} = pname; %#ok<AGROW>
        end
    end
end

% =========================================================================
% Utilities
% =========================================================================

function s = safeString(val, default)
    % Handle empty, missing, or invalid values safely
    if isempty(val)
        s = default;
    elseif isstring(val) && any(ismissing(val))
        s = default;
    elseif iscell(val) && isempty(val{1})
        s = default;
    else
        try
            s = char(string(val));
        catch
            s = default;
        end
    end
end
