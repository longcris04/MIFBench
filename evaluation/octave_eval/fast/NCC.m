function res=NCC(im1,im2)
% NCC  Vectorised copy of indexes/imageFusionMetrics-master/NCC.m (Octave speed-up).
%
%   The joint histogram the original fills pixel by pixel is counted with
%   accumarray (exact integer counts); everything after it is the original
%   code unchanged, so the result is the same bit for bit.  Used by
%   metricWang and Q_NCIE.

im1=double(im1);
im2=double(im2);

[hang,lie]=size(im1);
count=hang*lie;
N=256;
b=256;

h = accumarray([im1(:)+1, im2(:)+1], 1, [N N]);

h=h./sum(h(:));

im1_marg=sum(h);    % sum each column for im1
im2_marg=sum(h');   % sum each row for im2

H_x=-sum(im1_marg.*log2(im1_marg+(im1_marg==0)));
H_y=-sum(im2_marg.*log2(im2_marg+(im2_marg==0)));

H_xy=-sum(sum(h.*log2(h+(h==0))));

H_xy=H_xy/log2(b);
H_x=H_x/log2(b);
H_y=H_y/log2(b);

res=H_x+H_y-H_xy;
end
