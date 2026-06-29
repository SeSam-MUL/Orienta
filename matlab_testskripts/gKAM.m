function [gkam, gid, kam] = gKAM(ebsd,varargin)
% calculate grain average misorientation angle from KAM
% input:
%      ebsd (with grainID)
% 
%   options:
%      threshold:    angles to ignore (upper limit) in radian 
%      lowlim:       angles to ignore: (lower limit) in radian
%      order:        1st,2nd etc. neighbourhood around each point
% 
%   output:
%      gkam:         grain averaged kernel average misorientation angle
%                    for each grain unique(ebsd.grainId) in radian
%      gid:          grainId from ebsd
%      kam:          kernel average misorientation angle in radian
%
%   usage:
%   [grains,ebsd.grainId]=calcGrains(ebsd)
%   [gkam, gid, ebsd.prop.KAM] = gKAM(ebsd,'threshold',8*degree,'order',3)
%   plot(grains(gid),gkam./degree)
%


%   check if grainID exists
if isempty(ebsd.grainId)
     error('There is no ebsd.grainId. Run calcGrains first.')
end
%   get options
thresh = get_option(varargin,'threshold',8*degree);
order = get_option(varargin,'order',1);
lowlim= get_option(varargin,'lowlim',0);
% calc KAM according to options
kam=KAM(ebsd,'threshold',thresh,'order',order);
% set KAM below lowlim to nan
kam(kam<lowlim)=nan;
%find ebsd.grainID and index of corresponding kam
[gid,~,eindex] = unique(ebsd.grainId);
% calc the mean, ignoring nans
gkam = accumarray(eindex,kam(:),[],@nanmean);

if ~isempty(gkam) && isnan(gkam(1))
    gkam = gkam(2:end);
end

end