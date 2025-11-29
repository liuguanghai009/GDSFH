function x = window_reverse(windows, window_size, H, W)

%windows: Input tensor, size size window_szie,window_wize,C,B_

%return: x, size H W C B

B = floor(size(windows, 3) / (H * W / window_size / window_size));

x = reshape(windows, window_size, window_size,[], B, W / window_size, H / window_size);

x = permute(x, [2,6,1,5,3,4]);

x = reshape(x, H, W, [], B);

end