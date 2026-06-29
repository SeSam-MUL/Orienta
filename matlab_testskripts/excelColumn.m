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
