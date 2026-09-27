% gBC.m - MATLAB/MTEX reference scripts of Orienta (see README.md in this folder)
% Copyright (C) 2026 Dr. Irmgard Weissensteiner and Montanuniversitaet Leoben (ASCII spelling; see LICENSE)
% SPDX-License-Identifier: GPL-2.0-or-later
% This header was added for publication; the code below is as received.
function [gBCvals,gFe,gSi,gid] = gBC(ebsd)

if isempty(ebsd.grainId)
    error('There is no ebsd.grainId. Run calcGrains first.')
end

% --- IMPORTANT: work only on indexed points (recommended) ---
eb = ebsd('indexed');

% and ensure grainId > 0 (safety)
m = eb.grainId > 0;

gid_raw = eb.grainId(m);

% unique grain ids and mapping index
[gid,~,eindex] = unique(gid_raw);

% BC (make sure it is a vector aligned with m)
bc = eb.bc;
bc = bc(m);

gBCvals = accumarray(eindex, bc, [], @nanmean);

% Optional EDX channels (align with m)
if isfield(eb.prop,'Fe_Ka1')
    Fe = eb.prop.Fe_Ka1; Fe = Fe(m);
    gFe = accumarray(eindex, Fe, [], @nanmean);
else
    gFe = zeros(size(gBCvals));
end

if isfield(eb.prop,'Si_Ka1')
    Si = eb.prop.Si_Ka1; Si = Si(m);
    gSi = accumarray(eindex, Si, [], @nanmean);
else
    gSi = zeros(size(gBCvals));
end
end
