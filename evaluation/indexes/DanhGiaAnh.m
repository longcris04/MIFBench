% Ham nay thu hien danh gia 4 thong so cua Anh
% Trung binh, phuong sai, entropy, gradient 
function f = DanhGiaAnh(I)

%I = rgb2gray(I);
f(1) = mean(I(:));
f(2) = sqrt(var(I(:),1));
%f(2) = var(I(:),1);
f(3) = entropy(I);
f(4) = sharpness(I);
%f(5) = 
%f(5) = SpatialFrequency(I);
end

function f = sharpness(I)

[Gx, Gy] = imgradientxy(I,'intermediate');

S=sqrt(Gx.*Gx+Gy.*Gy);
f = sum(sum(S))./(numel(Gx));

end




