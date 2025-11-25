function T = parameterReactionTableExtended(model)
% Builds:
% Parameter, Reaction, ReactionRate, Rule, RuleType, OtherParameters, OtherSpeciesWithNotes (JSON)
% - Extracts both reactions AND rules that use each parameter
% - Species are compartment-qualified as "compartment.species".
% - Unqualified species tokens in ReactionRate/Rule map to all matching compartments.
% - Compartments referenced (in stoichiometry, rate, or rule) are ALSO included in the JSON.

    % ----- Index model-wide objects ----------------------------------------
    params          = sbioselect(model,'Type','parameter');
    speciesObjsAll  = sbioselect(model,'Type','species');
    compObjsAll     = sbioselect(model,'Type','compartment');
    rulesAll        = sbioselect(model,'Type','rule');

    allParamNames   = uniqstr({params.Name});
    [speciesByQualified, speciesByUnqualified] = indexSpecies(speciesObjsAll);
    compByName = indexCompartments(compObjsAll);     % name -> compartment object
    allCompNames = string(keys(compByName)).';

    % ----- Collect rows -----------------------------------------------------
    Parameter  = strings(0,1);
    Reaction   = strings(0,1);
    Rate       = strings(0,1);
    Rule       = strings(0,1);
    RuleType   = strings(0,1);
    OtherPars  = strings(0,1);
    OtherSpJS  = strings(0,1);

    for p = params(:)'
        [comps, ~] = findUsages(p);
        isRxn = arrayfun(@(c) isa(c,'SimBiology.Reaction'), comps);
        isRule = arrayfun(@(c) isa(c,'SimBiology.Rule'), comps);
        rxns  = comps(isRxn);
        rules = comps(isRule);


        for r = rxns(:)'
            rxnText = string(r.Reaction);
            rr = "";
            try, rr = string(r.ReactionRate); end

            ids = idsFromExpr(rr); % tokens (qualified and unqualified)

            % ---- Other parameters (robust regex search; allows comp.param) ----
            otherParams    = otherParamsInRate(rr, {params.Name}, p.Name);
            otherParamsStr = listString(string(otherParams));

            % ---- Species from stoichiometry (qualified) ----
            stoichQualified = strings(0,1);
            stoichComps     = strings(0,1);
            if ~isempty(r.Reactants)
                reactQ = arrayfun(@qualifiedName, r.Reactants(:), 'UniformOutput', false);
                stoichQualified = [stoichQualified; string(reactQ(:))]; %#ok<AGROW>
                stoichComps     = [stoichComps; arrayfun(@(x) string(x.Parent.Name), r.Reactants(:))]; %#ok<AGROW>
            end
            if ~isempty(r.Products)
                prodQ = arrayfun(@qualifiedName, r.Products(:), 'UniformOutput', false);
                stoichQualified = [stoichQualified; string(prodQ(:))]; %#ok<AGROW>
                stoichComps     = [stoichComps; arrayfun(@(x) string(x.Parent.Name), r.Products(:))]; %#ok<AGROW>
            end
            stoichComps = unique(stoichComps);

            % ---- Species from rate expression (qualified resolution) ----
            rateQualified = tokensToQualifiedSpecies(ids, speciesByQualified, speciesByUnqualified);

            % ---- Compartments referenced in the rate expression ----
            % 1) any "comp.spec" tokens → take the "comp" part
            % 2) any token that exactly equals a compartment name
            compsFromTokens = tokensToCompartments(ids, allCompNames);

            % Union of species involved
            allTheseQualified = uniqstr([cellstr(stoichQualified); cellstr(rateQualified)]);

            % Exclude any that literally equal parameter name
            allTheseQualified(strcmp(allTheseQualified, p.Name)) = [];

            % Gather species objects
            speciesList = [];
            for k = 1:numel(allTheseQualified)
                key = char(allTheseQualified{k});
                if isKey(speciesByQualified, key)
                    speciesList = [speciesList speciesByQualified(key)]; %#ok<AGROW>
                end
            end

            % Gather compartment objects (from stoichiometry + rate tokens)
            compNames = uniqstr([cellstr(stoichComps); cellstr(compsFromTokens)]);
            compList = [];
            for k = 1:numel(compNames)
                cname = char(compNames{k});
                if isKey(compByName, cname)
                    compList = [compList compByName(cname)]; %#ok<AGROW>
                end
            end

            % Build JSON: species + compartments, with requested schema
            otherSpeciesJSON = componentsNotesJSON(speciesList, compList);

            % Append row
            Parameter(end+1,1)  = string(p.Name);
            Reaction(end+1,1)   = rxnText;
            Rate(end+1,1)       = rr;
            Rule(end+1,1)       = "";
            RuleType(end+1,1)   = "";
            OtherPars(end+1,1)  = otherParamsStr;
            OtherSpJS(end+1,1)  = otherSpeciesJSON;
        end

        % ----- Process rules for this parameter -----
        for rule = rules(:)'
            ruleText = "";
            ruleTypeText = "";
            ruleExpr = "";

            try
                ruleText = string(rule.Rule);
                ruleTypeText = string(rule.RuleType);
                ruleExpr = ruleText;
            catch
                % Skip if rule properties can't be accessed
                continue;
            end

            ids = idsFromExpr(ruleExpr); % tokens (qualified and unqualified)

            % ---- Other parameters (robust regex search; allows comp.param) ----
            otherParams    = otherParamsInRate(ruleExpr, {params.Name}, p.Name);
            otherParamsStr = listString(string(otherParams));

            % ---- Species from rule expression (qualified resolution) ----
            ruleQualified = tokensToQualifiedSpecies(ids, speciesByQualified, speciesByUnqualified);

            % ---- Compartments referenced in the rule expression ----
            compsFromTokens = tokensToCompartments(ids, allCompNames);

            % Convert to cell array for consistent processing (like reactions)
            allTheseQualified = uniqstr(cellstr(ruleQualified));

            % Exclude any that literally equal parameter name
            allTheseQualified(strcmp(allTheseQualified, p.Name)) = [];

            % Gather species objects
            speciesList = [];
            for k = 1:numel(allTheseQualified)
                key = char(allTheseQualified{k});
                if isKey(speciesByQualified, key)
                    speciesList = [speciesList speciesByQualified(key)]; %#ok<AGROW>
                end
            end

            % Gather compartment objects
            compNames = cellstr(compsFromTokens);
            compList = [];
            for k = 1:numel(compNames)
                cname = char(compNames{k});
                if isKey(compByName, cname)
                    compList = [compList compByName(cname)]; %#ok<AGROW>
                end
            end

            % Build JSON: species + compartments, with requested schema
            otherSpeciesJSON = componentsNotesJSON(speciesList, compList);

            % Append row
            Parameter(end+1,1)  = string(p.Name);
            Reaction(end+1,1)   = "";
            Rate(end+1,1)       = "";
            Rule(end+1,1)       = ruleText;
            RuleType(end+1,1)   = ruleTypeText;
            OtherPars(end+1,1)  = otherParamsStr;
            OtherSpJS(end+1,1)  = otherSpeciesJSON;
        end
    end

    % ----- Build table, dedupe, sort ---------------------------------------
    T = table(Parameter, Reaction, Rate, Rule, RuleType, OtherPars, OtherSpJS, ...
        'VariableNames', {'Parameter','Reaction','ReactionRate','Rule','RuleType','OtherParameters','OtherSpeciesWithNotes'});

    if ~isempty(T)
        T = unique(T, 'rows');
        T = sortrows(T, {'Parameter','Reaction','Rule'});
    end
end

% ================= helpers =====================

function out = uniqstr(c)
    if isstring(c); c = cellstr(c); end
    out = unique(c(:));
end

function q = qualifiedName(specObj)
    % Return "compartment.species"
    comp = "";
    try, comp = string(specObj.Parent.Name); end
    q = string(comp + "." + string(specObj.Name));
end

function ids = idsFromExpr(expr)
    % Match identifiers with optional single dot: comp.spec OR spec (single dot max)
    if strlength(expr)==0
        ids = string.empty(0,1); return
    end
    % Use the same proven pattern from otherParamsInRate function
    exprChar = char(expr);

    % Pattern based on otherParamsInRate: (?<![A-Za-z0-9_])(?:[A-Za-z_]\w*\.)?IDENTIFIER(?![A-Za-z0-9_])
    % Modified to capture all identifiers, including compartment.species pairs
    pattern = '(?<![A-Za-z0-9_])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)(?![A-Za-z0-9_])';
    m = regexp(exprChar, pattern, 'match');

    if isempty(m)
        ids = string.empty(0,1);
    else
        ids = string(unique(m));
    end
end

function names = otherParamsInRate(rateExpr, allParamNames, focalName)
    % Robust: match either "name" or "comp.name" with identifier boundaries.
    names = string.empty(0,1);
    if strlength(string(rateExpr)) == 0 || isempty(allParamNames), return; end
    rateExpr = char(string(rateExpr));
    allParamNames = string(allParamNames(:));
    allParamNames = allParamNames(allParamNames ~= string(focalName));
    hit = false(numel(allParamNames),1);
    for i = 1:numel(allParamNames)
        nm = char(allParamNames(i));
        pat = ['(?<![A-Za-z0-9_])(?:[A-Za-z_]\w*\.)?' , nm , '(?![A-Za-z0-9_])'];
        if ~isempty(regexp(rateExpr, pat, 'once'))
            hit(i) = true;
        end
    end
    names = allParamNames(hit);
end

function [byQualified, byUnqualified] = indexSpecies(speciesObjs)
    % Maps:
    %  - byQualified: "comp.spec" -> array of species objects
    %  - byUnqualified: "spec" -> array of species objects (potentially many comps)
    byQualified   = containers.Map('KeyType','char','ValueType','any');
    byUnqualified = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(speciesObjs)
        q = char(qualifiedName(speciesObjs(i)));
        u = char(string(speciesObjs(i).Name));
        if isKey(byQualified, q),   byQualified(q)   = [byQualified(q) speciesObjs(i)];
        else,                       byQualified(q)   = speciesObjs(i);
        end
        if isKey(byUnqualified, u), byUnqualified(u) = [byUnqualified(u) speciesObjs(i)];
        else,                       byUnqualified(u) = speciesObjs(i);
        end
    end
end

function compByName = indexCompartments(compObjs)
    compByName = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(compObjs)
        nm = char(string(compObjs(i).Name));
        if isKey(compByName, nm), compByName(nm) = [compByName(nm) compObjs(i)];
        else,                     compByName(nm) = compObjs(i);
        end
    end
end

function qualified = tokensToQualifiedSpecies(tokens, byQ, byU)
    % Convert tokens into qualified species names.
    qualified = strings(0,1);
    for t = tokens(:)'
        tok = char(t);
        if contains(tok, '.')
            if isKey(byQ, tok)
                qualified(end+1,1) = string(tok); %#ok<AGROW>
            end
        else
            if isKey(byU, tok)
                objs = byU(tok);
                qnames = arrayfun(@qualifiedName, objs, 'UniformOutput', false);
                qualified = [qualified; string(qnames(:))]; %#ok<AGROW>
            end
        end
    end
    qualified = string(uniqstr(cellstr(qualified)));
end

function compNames = tokensToCompartments(tokens, allCompNames)
    % Extract compartments from tokens:
    % - any "comp.spec" -> "comp"
    % - any token exactly equal to a compartment name -> include it
    compNames = strings(0,1);
    toks = string(tokens(:));
    % From qualified tokens
    hasDot = contains(toks, ".");
    if any(hasDot)
        left = extractBefore(toks(hasDot), ".");
        compNames = [compNames; left];
    end
    % Exact matches to compartment names
    exactComp = toks(ismember(toks, allCompNames));
    compNames = string(uniqstr([cellstr(compNames); cellstr(exactComp)]));
end

function s = componentsNotesJSON(specObjs, compObjs)
    % Build JSON array with fields:
    %   - name: species as "comp.spec", compartments as "comp"
    %   - compartment: species -> comp; compartments -> comp name
    %   - notes
    rows = struct('name', {}, 'compartment', {}, 'notes', {});
    % Species entries
    for i = 1:numel(specObjs)
        comp  = "";
        try, comp = string(specObjs(i).Parent.Name); end
        nmQ   = comp + "." + string(specObjs(i).Name);
        notes = "";
        try, notes = string(specObjs(i).Notes); end
        rows(end+1) = struct('name', nmQ, 'compartment', comp, 'notes', notes); %#ok<AGROW>
    end
    % Compartment entries
    for i = 1:numel(compObjs)
        comp  = string(compObjs(i).Name);
        notes = "";
        try, notes = string(compObjs(i).Notes); end
        rows(end+1) = struct('name', comp, 'compartment', comp, 'notes', notes); %#ok<AGROW>
    end
    % Deduplicate by 'name'
    [~, iu] = unique(string({rows.name})');
    rows = rows(sort(iu));
    s = string(jsonencode(rows));
end

function s = listString(strs)
    % Format a string array as a Python-like list string
    % e.g. ["a","b"] -> "['a','b']"
    if isempty(strs)
        s = "[]";
    else
        strs = string(strs(:));
        s = "[" + join("'" + strs + "'", ",") + "]";
        s = string(s);
    end
end
