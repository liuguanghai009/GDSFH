function attn_mask = AM (input_resolution,window_size,shift_size)
% Calculate attention mask for SW-MSA

H = input_resolution(1);

W = input_resolution(2);

img_mask = zeros(H, W, 1, 1);  % H W 1 1

% 这里这样取的原因是窗口移位后，为了保证窗口之间进行信息传递，且进行自注意力机制。

h_slices = {1:(H-window_size), (H-window_size+1):(H-shift_size), (H-shift_size+1):H};

w_slices = {1:(W-window_size), (W-window_size+1):(W-shift_size), (W-shift_size+1):W};

cnt = 0;

% 特征图分为9块并且标号 0-8

for h = h_slices

    for w = w_slices

        img_mask( h{:}, w{:}, :, :) = cnt;

        cnt = cnt + 1;

    end

end

% 先按照窗口尺寸为 window_size 进行分块

mask_windows = window_partition(img_mask, window_size);  % window_size * window_size, 1, nW*1

mask_windows = permute(mask_windows,[1,3,2]); % window_size * window_size, nW*1

attn_mask = bsxfun(@minus,permute(mask_windows,[3,1,2]), permute(mask_windows,[1,3,2])); % 这里的作用是将窗口的 index 相减， 仿照自注意力，也是一种 Broadcasting 操作

attn_mask(attn_mask ~= 0) = -100.0; % 不同窗口的 index 相减不为零，则赋值 -100， 另其在softmax时候为0

attn_mask(attn_mask == 0) = 0.0;

end