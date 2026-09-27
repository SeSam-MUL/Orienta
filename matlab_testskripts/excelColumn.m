% excelColumn.m - MATLAB/MTEX reference scripts of Orienta (see README.md in this folder)
% Copyright (C) 2026 Dr. Irmgard Weissensteiner and Montanuniversitaet Leoben (ASCII spelling; see LICENSE)
% SPDX-License-Identifier: GPL-2.0-or-later
% This header was added for publication; the code below is as received.
function col = excelColumn(k)
%EXCELCOLUMN Convert 1-based column index to Excel column letters.
%   1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA, ...

    if k < 1 || k ~= floor(k)
        error("excelColumn: k must be a positive integer.");
    end
    col = "";
    while k > 0
        r = mod(k-1, 26);
        col = char('A' + r) + col;
        k = floor((k-1)/26);
    end
end
