function [ebsd_smoothedFill, grains, oM1, r, step, ebsd1, maxX, maxY, ebsd1plot] = ...
    ImportAndModify_MTEX6(formatstring, geomRot, geomRotAngle, rec, tc, ...
                    particleRem, datname, foldername,  ...
                    gbThreshold, num_pixel, r1, r2, r3, n)
% ImportAndModify  Load, crop, denoise and reconstruct grains from EBSD data.
%
%  Workflow:
%    1. Load  →  2. Crop  →  3. tilt/geom correction
%    4. EDX map (optional)  →  5. calcGrains (initial)
%    6. Particle removal (EDX, vectorised)  →  7. HalfQuadratic denoising
%    8. Rotation/alignment  →  9. Final calcGrains  →  10. Documentation

%% ── 0. Crystal symmetry & phase label ───────────────────────────────────
CS  = crystalSymmetry('m-3m', [4.05 4.05 4.05], 'mineral', 'Aluminium');
ph  = 'indexed';                  % default: use all indexed points
if particleRem == 1
    ph = 'Aluminium';
end

%% ── 1. Load data ─────────────────────────────────────────────────────────
ebsd1 = loadEBSD_h5oina([foldername datname formatstring], ...
    'cs', CS, 'convertEuler2SpatialReferenceFrame');
cd(datname);

%% ── 2. Crop ──────────────────────────────────────────────────────────────
% Option 1 (active): centre crop – 15 % of map width/height
% To switch options: comment Option 1, uncomment the desired block.
condition = inpolygon(ebsd1, [0.5*max(ebsd1.x),  0.5*max(ebsd1.y), ...
                               0.15*max(ebsd1.x), 0.15*max(ebsd1.y)]);
ebsd1 = ebsd1(condition);

% Option 2a – click-to-select single point (interactive, for test runs):
%   plot(ebsd1, ebsd1.bc); colormap gray;
%   e = selectPoint(ebsd1);
%   condition = inpolygon(ebsd1, [(ex(n)-200) ey(n) 200 200]);
%   ebsd1 = ebsd1(condition);

% Option 2b – interactive freehand region:
%   ebsd1 = selectInteractive(ebsd1);

% Option 3 – reduce resolution (fast texture checks):
%   ebsd1 = reduce(ebsd1);

% Option 4 – polygon selection:
%   figtemp = figure; plot(ebsd1, ebsd1.bc);
%   poly    = selectPolygon;
%   ebsd1   = ebsd1(inpolygon(ebsd1, poly));
%   close(figtemp);

%% ── 3. Geometric & tilt corrections ─────────────────────────────────────
% Geometric rotation (for display only → ebsd1plot)
if geomRot == 1
    ebsd1plot = rotate(ebsd1, ...
        rotation.byAxisAngle(zvector, geomRotAngle(n)*degree), 'keepEuler');
    ebsd1plot = ebsd1plot.gridify;
else
    ebsd1plot = ebsd1;
end

% 70° tilt correction (SEM stage tilt)
if tc == 1
    fac           = 1 / cos(70*degree);
    ebsd1.y       = fac * ebsd1.y;
    ebsd1.unitCell(:,2) = fac * ebsd1.unitCell(:,2);
end

% Save raw (cropped + corrected) data – after crop to keep file small
save(['variables' datname '.mat'], 'ebsd1');

%% ── 4. Shear / rectification correction (interactive) ───────────────────
if rec == 1
    plot(ebsd1, ebsd1.bc);
    disp('Select 4 sample corners (TL → BL → BR → TR). Click = waitforbuttonpress.');
    ax = gca;
    waitforbuttonpress; p1 = get(ax, 'CurrentPoint');
    waitforbuttonpress; p2 = get(ax, 'CurrentPoint');
    waitforbuttonpress; p3 = get(ax, 'CurrentPoint');
    waitforbuttonpress; p4 = get(ax, 'CurrentPoint');

    deltay12 = p1(1,2) - p2(1,2);
    deltay43 = p4(1,2) - p3(1,2);
    k = (deltay12 - deltay43) / (p2(1,2) - p3(1,2));
    d = deltay43 - p3(1,2) * k;

    ebsdcorr   = ebsd1;
    ebsdcorr.y = ebsdcorr.y + (ebsdcorr.x / max(ebsdcorr.x)) .* (k*ebsd1.y + d);
    ebsdcorr.unitCell = calcUnitCell([ebsdcorr.x(:), ebsdcorr.y(:)]);

    plot(ebsdcorr(ph), ebsdcorr(ph).orientations);
    ebsd1 = ebsdcorr;
    clear ebsdcorr;
end

maxX = max(ebsd1.x(:));
maxY = max(ebsd1.y(:));

%% ── 5. Gridify & stepsize ────────────────────────────────────────────────
ebsd1 = ebsd1.gridify;
step  = 2 * ebsd1.unitCell(1).x;

%% ── 6. EDX chemistry map (only for fine-step maps, step < 2.5 µm) ───────
if step < 2.5
    try
        useKa      = isfield(ebsd1.prop, 'Al_Ka1');
        useKseries = isfield(ebsd1.prop, 'Al_K_series');

        if useKa || useKseries
            chemFig = newMtexFigure;

            if useKa
                elements_Ka = { ...
                    'Al_Ka1',   'Al'; ...
                    'Fe_Ka1',   'Fe'; ...
                    'Mn_Ka1',   'Mn'; ...
                    'Mg_Ka1_2', 'Mg'; ...
                    'Zn_Ka1',   'Zn'; ...
                    'Cu_Ka1',   'Cu'; ...
                    'Cu_La1_2', 'Cu (L)'; ...
                    'Cr_Ka1',   'Cr'; ...
                    'Si_Ka1',   'Si'};
                for k = 1:size(elements_Ka, 1)
                    fname = elements_Ka{k,1};
                    if isfield(ebsd1.prop, fname)
                        plot(ebsd1, ebsd1.prop.(fname));
                        mtexColorMap inferno;
                        setColorRange('tight');
                        mtexTitle(elements_Ka{k,2});
                        if k < size(elements_Ka, 1); nextAxis; end
                    end
                end

            else  % useKseries
                elements_Ks = { ...
                    'Fe_K_series',  'Fe'; ...
                    'Mn_K_series',  'Mn'; ...
                    'Si_K_series',  'Si'; ...
                    'Mg_K_series2', 'Mg'; ...
                    'Zn_La1_2',     'Zn'; ...
                    'Cu_La1_2',     'Cu'; ...
                    'Ag_La1',       'Ag'; ...
                    'Al_Ka1',       'Al'};
                for k = 1:size(elements_Ks, 1)
                    fname = elements_Ks{k,1};
                    if isfield(ebsd1.prop, fname)
                        plot(ebsd1, ebsd1.prop.(fname));
                        mtexColorMap inferno;
                        setColorRange('tight');
                        mtexTitle(elements_Ks{k,2});
                        if k < size(elements_Ks, 1); nextAxis; end
                    end
                end
            end

            mtexColorMap magma;
            mtexColorbar;
            saveFigure(['chemData' datname '.png']);
        end
    catch ME
        warning('EDX map failed: %s', ME.message);
    end
end
close all;

%% ── 7. Initial grain reconstruction (for particle ID) ───────────────────
ebsd_good = ebsd1;
[grainsraw, ebsd_good.grainId] = calcGrains(ebsd_good, ...
    'alpha', 2.0, 'angle', 5*degree);
grainsraw = smooth(grainsraw, 4);
ebsd_good = ebsd_good.gridify;
disp('calcGrains 1 finished');

%% ── 8. Particle identification & removal (EDX-based, vectorised) ─────────
if particleRem == 1

    CS_par = { 'notIndexed', ...
        crystalSymmetry('m-3m', [4 4 4], ...
            'mineral', 'Aluminium', 'color', [0.53 0.81 0.98]), ...
        crystalSymmetry('m-3', [12.65 12.65 12.65], ...
            'mineral', 'Alpha',     'color', [1.00 0.80 0.00]) };
    CSA = CS_par{1,2};
    CST = CS_par{1,3};
    h   = Miller({1,0,0},{1,1,0},{1,1,1}, CSA);

    useKa      = isfield(ebsd_good.prop, 'Fe_Ka1');
    useKseries = isfield(ebsd_good.prop, 'Fe_K_series');

    if useKa || useKseries
        disp('Particle identification: vectorised grain-average EDX …');
        tic

        % Vectorised grain-average: accumarray over all indexed EBSD points
        m   = ebsd_good.grainId > 0;
        eb  = ebsd_good(m);
        [gid_unique, ~, eidx] = unique(eb.grainId);
        nG  = numel(gid_unique);

        if useKa
            gFe = accumarray(eidx, double(eb.prop.Fe_Ka1), [nG 1], @mean, NaN);
            gAl = accumarray(eidx, double(eb.prop.Al_Ka1), [nG 1], @mean, NaN);
            gMg = condAccum(eb, eidx, nG, 'Mg_Ka1_2');
            gSi = condAccum(eb, eidx, nG, 'Si_Ka1');
            Fe_thresh = 3300;  Al_thresh = 220000;  Si_thresh = 200000;
        else
            gFe = accumarray(eidx, double(eb.prop.Fe_K_series), [nG 1], @mean, NaN);
            gAl = accumarray(eidx, double(eb.prop.Al_K_series), [nG 1], @mean, NaN);
            gSi = condAccum(eb, eidx, nG, 'Si_K_series');
            gMg = nan(nG, 1);
            Fe_thresh = 1850;  Al_thresh = 220000;  Si_thresh = Inf;
        end

        % Map accumarray results back onto grainsraw via grain IDs
        [~, loc] = ismember(gid_unique, grainsraw.id);
        valid = loc > 0;
        grainsraw(loc(valid)).prop.Fe = gFe(valid);
        grainsraw(loc(valid)).prop.Al = gAl(valid);
        grainsraw(loc(valid)).prop.Mg = gMg(valid);
        grainsraw(loc(valid)).prop.Si = gSi(valid);
        toc

        try
            % Fe-rich particles
            Fegrainsraw = grainsraw(grainsraw.prop.Fe > Fe_thresh & ...
                                    grainsraw.prop.Al < Al_thresh);
            % Si-rich oxides (Ka1 branch only)
            if isfinite(Si_thresh) && isfield(grainsraw.prop, 'Si')
                Fegrainsraw = [Fegrainsraw ...
                    grainsraw(grainsraw.prop.Si > Si_thresh)]; %#ok<AGROW>
            end

            Fegrainsraw = smooth(Fegrainsraw, 4);
            Fegrainsraw(Fegrainsraw.equivalentRadius < 1.2).CS = CST;
            ebsd_good(Fegrainsraw).CS = CST;

            % Remove hollow / needle-like artifacts
            % log(area/perimeter) < 0.2 → not a compact grain
            toRemove = Fegrainsraw( ...
                log(Fegrainsraw.numPixel ./ Fegrainsraw.boundarySize) < 0.2);
            ebsd_good(toRemove) = [];

            fprintf('  Particles assigned: %d  |  removed (holes/needles): %d\n', ...
                length(Fegrainsraw), length(toRemove));
        catch ME
            warning('Particle classification failed: %s', ME.message);
        end

    else
        disp('No EDX data found – particle removal skipped.');
    end

    % Low-BC cleanup (unreliable indexing near particles / boundaries)
    ebsd_good(ebsd_good.bc < 28) = [];

    % NaN rotations from gridify of empty regions → mark as notIndexed
    try
        ebsd_good(isnan(ebsd_good.rotations)).phaseId = 1;
    catch; end

    % Grain reconstruction after particle removal
    disp('calcGrains after particle removal …');
    [grainsraw, ebsd_good.grainId] = calcGrains(ebsd_good, ...
        'alpha', 2.0, 'angle', 5*degree);
    ebsd_good = ebsd_good.gridify;
    disp('calcGrains after particle removal – done');

end  % particleRem

%% ── 9. HalfQuadratic denoising (tiled, memory-safe) ─────────────────────
ebsd0    = ebsd_good;
[ny, nx] = size(ebsd0);

F           = halfQuadraticFilter;
F.alpha     = 0.7;
F.threshold = 7.5*degree;
F.tol       = 0.1*degree;
F.iterMax   = 100;

% Halo: at least 50 px, or 3× the largest grain radius in pixels
halo = max(50, round(3 * max(grainsraw.equivalentRadius) / step));
xMid = round(nx / 2);

parts = { ...
    struct('x0',1,      'x1',xMid, 'xh0',1,                  'xh1',min(nx,xMid+halo)); ...
    struct('x0',xMid+1, 'x1',nx,   'xh0',max(1,xMid+1-halo), 'xh1',nx) };

ebsd_smoothed = ebsd0;   % output container – preserves full grid

for p = 1:numel(parts)
    fprintf('Denoising part %d/%d …\n', p, numel(parts));
    x0  = parts{p}.x0;   x1  = parts{p}.x1;
    xh0 = parts{p}.xh0;  xh1 = parts{p}.xh1;

    % ── FIX: extract tile via 2-D matrix indexing, NOT linear + reshape ──
    % ebsd0 is EBSDsquare → direct 2D subscript indexing preserves the grid
    ebsd_tile = ebsd0(:, xh0:xh1);          % [ny × (xh1-xh0+1)] tile, stays gridded
    % ─────────────────────────────────────────────────────────────────────

    % Work on indexed points only
    ebsd_tile_idx = ebsd_tile('indexed');

    % Local grain reconstruction on the tile
    [gr_tile, ebsd_tile_idx.grainId] = calcGrains(ebsd_tile_idx, ...
        'angle', 5*degree, 'minPixel', 3);

    % Smooth + fill non-indexed points within grains
    ebsd_tile_s = smooth(ebsd_tile_idx, F, 'fill', gr_tile);

    % ── Write back only the CORE columns (x0..x1) ────────────────────────
    % After smooth/fill, ebsd_tile_s is ungridded → regridify onto tile shape
    ebsd_tile_s = ebsd_tile_s.gridify;

    % Column offset of core within the halo tile (1-based)
    coreStart = x0 - xh0 + 1;
    coreEnd   = x1 - xh0 + 1;

    % Write rotations from core columns of smoothed tile back into full grid
    ebsd_smoothed(:, x0:x1).rotations = ebsd_tile_s(:, coreStart:coreEnd).rotations;

    fprintf('  Part %d done. Core x=[%d..%d], Halo x=[%d..%d]\n', ...
        p, x0, x1, xh0, xh1);
end

ebsd_smoothedFill = ebsd_smoothed;


%% ── 10. IPF maps – raw data (3 directions) ───────────────────────────────
oM1 = ipfTSLKey(ebsd1(ph));
ori1 = ebsd1(ph).orientations;

figtemp = figure;
bcData  = ebsd1.bc;

directions = {xvector, 'X'; yvector, 'Y'; zvector, 'Z'};
for d = 1:3
    if d > 1; nextAxis; end
    plot(ebsd1, bcData);
    mtexColorMap black2white;
    setColorRange('tight');
    hold on;
    oM1.inversePoleFigureDirection = directions{d,1};
    plot(ebsd1(ph), oM1.orientation2color(ori1), 'FaceAlpha', 0.7);
    mtexTitle(['IPF – ' directions{d,2}]);
end
saveas(figtemp, ['RawDataWithoutBadPts_' datname '.png']);
close all;

%% ── 11. Orientation rotation / alignment ────────────────────────────────
rot1 = rotation.byAxisAngle(zvector, r1*degree);
rot2 = rotation.byAxisAngle(xvector, r2*degree);
rot3 = rotation.byAxisAngle(yvector, r3*degree);
r    = rot3 * rot2 * rot1;

geoRot = rotation.byAxisAngle(zvector, geomRotAngle(n)*degree);

if geomRot == 0
    ebsd_smoothedFill = rotate(ebsd_smoothedFill, r, 'keepXY');
    ebsd_smoothedFill = ebsd_smoothedFill.gridify;
elseif geomRot == 1
    ebsd_smoothedFill = rotate(ebsd_smoothedFill, r, 'keepXY');
    ebsd_smoothedFill = rotate(ebsd_smoothedFill, geoRot, 'keepEuler');
    ebsd_smoothedFill = ebsd_smoothedFill.gridify;
end

%% ── 12. Final grain reconstruction (on smoothed, rotated data) ───────────
disp('calcGrains final …');
[grains, ebsd_smoothedFill.grainId] = calcGrains(ebsd_smoothedFill, ...
    'alpha', 2, 'angle', [gbThreshold, 2*degree]);
grains = smooth(grains, 4);
disp('calcGrains final – done');

%% ── 13. Documentation ────────────────────────────────────────────────────
TxInfo = {'Basic Information:'; 'stepsize'; 'max. X'; 'max. Y'; ...
    'Number of good datapoints'; 'rotation phi1'; 'rotation Phi'; ...
    'rotation phi2'; 'min. Pixel Grain'; 'Grain Boundary Threshold'; ...
    'number grains'};
DataRow = [{datname}; step; maxX; maxY; length(ebsd_smoothedFill(ph)); ...
    r.phi1/degree; r.Phi/degree; r.phi2/degree; ...
    num_pixel; gbThreshold/degree; length(grains(ph))];

docFile = [foldername 'Documentation.xlsx'];
writetable(cell2table(TxInfo.'),  docFile, 'WriteVariableNames', 0, 'Range', 'A1:K1');
writetable(cell2table(DataRow.'), docFile, 'WriteVariableNames', 0, ...
    'Range', ['A' num2str(n+1) ':K' num2str(n+1)]);

end  % ImportAndModify


%% ── Local helper: conditional accumarray for optional EDX fields ─────────
function g = condAccum(eb, eidx, nG, fieldname)
% Returns grain-average of eb.prop.(fieldname), or NaN-vector if absent.
    if isfield(eb.prop, fieldname)
        g = accumarray(eidx, double(eb.prop.(fieldname)), [nG 1], @mean, NaN);
    else
        g = nan(nG, 1);
    end
end
