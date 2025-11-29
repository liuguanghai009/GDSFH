function relative_position_index = RPI (window_size)

% 根据相对位置索引 relative_position_index 查 relative_position_bias_table 得到相对位置偏执参数

% Create coordinates along the height and width dimensions

coords_h = 0:window_size-1;

coords_w = 0:window_size-1;

% Create meshgrid for the coordinates

[coords_w, coords_h] = meshgrid(coords_h, coords_w);

coords = cat(3, coords_h, coords_w);

coords = permute(coords,[2,1,3]);

% Flatten the coordinates

coords_flatten = reshape(coords, [], 2);

% Calculate relative coordinates

relative_coords_h = coords_flatten(:, 1) - coords_flatten(:, 1)'; % 行相对位置，每一个元素对周边所有

relative_coords_w = coords_flatten(:, 2) - coords_flatten(:, 2)'; % 列相对位置，每一个元素对周边所有

relative_coords = cat(3,relative_coords_h, relative_coords_w);

% Shift relative coordinates to start from 0

relative_coords(:, :, 1) = relative_coords(:, :, 1) + window_size - 1; %防止有负数

relative_coords(:, :, 2) = relative_coords(:, :, 2) + window_size - 1; %防止有负数

% Calculate relative position index
relative_coords(:, :, 1) = relative_coords(:, :, 1) .* (2 * window_size - 1); % 将所有行标乘以窗口列大小 (2*window_size -1)

relative_position_index = sum(relative_coords, 3)+1;  % 将行标与列标相加。 Wh*Ww, Wh*Ww

end