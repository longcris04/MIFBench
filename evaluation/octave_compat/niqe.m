function score = niqe(im, mu_prisparam, cov_prisparam, sharpness_threshold)
% NIQE  Naturalness Image Quality Evaluator (no-reference).
%
%   SCORE = NIQE(IM) scores the distortion of IM against a multivariate
%   Gaussian model fitted to pristine natural images. Lower is better.
%
%   Octave replacement for MATLAB's Image Processing Toolbox niqe(), which
%   does not exist in Octave. This implements the original algorithm of
%   Mittal, Soundararajan & Bovik, "Making a Completely Blind Image Quality
%   Analyzer", IEEE Signal Processing Letters 20(3), 2013, using the
%   authors' published pristine model (niqe_modelparameters.mat).
%
%   Values are close to, but NOT identical with, MATLAB's niqe(): MathWorks
%   fitted their own pristine model on a different image set. Score every
%   image of a comparison with the same implementation; never mix NIQE
%   numbers produced here with numbers produced by MATLAB.
%
%   IM must be at least 96x96. Integer images are taken on their native
%   0-255 scale; double images are assumed to lie in [0,1], as in MATLAB.

  persistent mu_pris cov_pris

  if nargin < 4 || isempty(sharpness_threshold)
    sharpness_threshold = 0.75;   % as published; set -Inf to keep every patch
  end

  if nargin < 3
    if isempty(mu_pris)
      here = fileparts(mfilename('fullpath'));
      d = load(fullfile(here, 'niqe_modelparameters.mat'));
      mu_pris = d.mu_prisparam(:)';
      cov_pris = d.cov_prisparam;
    end
    mu_prisparam = mu_pris;
    cov_prisparam = cov_pris;
  end

  blocksizerow = 96;
  blocksizecol = 96;

  if ndims(im) == 3
    im = rgb2gray(im);
  end

  if isinteger(im)
    im = double(im);
  else
    im = double(im);
    if max(im(:)) <= 1
      im = im * 255;
    end
  end

  [row, col] = size(im);
  block_rownum = floor(row / blocksizerow);
  block_colnum = floor(col / blocksizecol);
  if block_rownum < 1 || block_colnum < 1
    error('niqe: image is %dx%d, needs to be at least %dx%d', ...
          row, col, blocksizerow, blocksizecol);
  end
  im = im(1:block_rownum*blocksizerow, 1:block_colnum*blocksizecol);

  window = fspecial('gaussian', 7, 7/6);
  window = window / sum(window(:));

  feat = [];
  sharpness = [];

  for itr_scale = 1:2
    mu        = imfilter(im, window, 'replicate');
    sigma     = sqrt(abs(imfilter(im.*im, window, 'replicate') - mu.*mu));
    structdis = (im - mu) ./ (sigma + 1);

    brow = blocksizerow / itr_scale;
    bcol = blocksizecol / itr_scale;
    nr   = floor(size(structdis, 1) / brow);
    nc   = floor(size(structdis, 2) / bcol);

    feat_scale = zeros(nr*nc, 18);
    if itr_scale == 1
      sharpness = zeros(nr*nc, 1);
    end

    % Column-major over blocks, matching how the original blkproc output is
    % reshaped -- feat rows and the sharpness mask must stay aligned.
    k = 0;
    for j = 1:nc
      for i = 1:nr
        k = k + 1;
        rr = (i-1)*brow+1 : i*brow;
        cc = (j-1)*bcol+1 : j*bcol;
        feat_scale(k,:) = computefeature(structdis(rr, cc))';
        if itr_scale == 1
          blk_sigma = sigma(rr, cc);
          sharpness(k) = mean(blk_sigma(:));
        end
      end
    end

    feat = [feat, feat_scale];
    im = imresize(im, 0.5);
  end

  distparam = feat(sharpness > sharpness_threshold, :);  % sharp patches only
  distparam = distparam(~any(isnan(distparam), 2), :);

  if size(distparam, 1) < 2
    warning('niqe: only %d usable patch(es); returning NaN', size(distparam,1));
    score = NaN;
    return;
  end

  mu_distparam  = mean(distparam, 1);
  cov_distparam = cov(distparam);

  invcov_param = pinv((cov_prisparam + cov_distparam) / 2);
  d = mu_prisparam - mu_distparam;
  score = sqrt(d * invcov_param * d');
end

function feat = computefeature(structdis)
% 18-D feature vector of one patch: an AGGD fit of the MSCN coefficients
% plus AGGD fits of their four neighbouring-pair products.

  feat = [];
  [alpha, betal, betar] = estimateaggdparam(structdis(:));
  feat = [feat; alpha; (betal+betar)/2];

  shifts = [0 1; 1 0; 1 1; 1 -1];
  for itr_shift = 1:4
    shifted_structdis = circshift(structdis, shifts(itr_shift,:));
    pair = structdis(:) .* shifted_structdis(:);
    [alpha, betal, betar] = estimateaggdparam(pair);
    meanparam = (betar - betal) * (gamma(2/alpha) / gamma(1/alpha));
    feat = [feat; alpha; meanparam; betal; betar];
  end
end

function [alpha, betal, betar] = estimateaggdparam(vec)
% Moment-match an asymmetric generalized Gaussian to VEC.

  persistent gam r_gam
  if isempty(gam)
    gam   = 0.2:0.001:10;
    r_gam = ((gamma(2./gam)).^2) ./ (gamma(1./gam) .* gamma(3./gam));
  end

  leftstd  = sqrt(mean((vec(vec < 0)).^2));
  rightstd = sqrt(mean((vec(vec > 0)).^2));

  gammahat  = leftstd / rightstd;
  rhat      = (mean(abs(vec)))^2 / mean(vec.^2);
  rhatnorm  = (rhat * (gammahat^3 + 1) * (gammahat + 1)) / ((gammahat^2 + 1)^2);

  [~, array_position] = min((r_gam - rhatnorm).^2);
  alpha = gam(array_position);

  betal = leftstd  * sqrt(gamma(1/alpha) / gamma(3/alpha));
  betar = rightstd * sqrt(gamma(1/alpha) / gamma(3/alpha));
end
