function windows = window_partition(x, window_size)

% x: Input tensor, size H x W x C x B

% Get the dimensions of the input tensor

[H, W, C, B] = size(x);
%------------------------第一种方法，比较形象-------------------------------
% Reshape the input tensor to create non-overlapping windows

% windows = mat2cell(x, window_size*ones(1, H/window_size), window_size*ones(1, W/window_size), C, B);

% windows = permute(windows,[2,1]);

% Convert the cell array of windows to a 4D array

% windows = cat(4, windows{:});

% windows = permute(windows,[2,1,3,4]);

% windows = reshape(windows,window_size*window_size,C,[]);
%----------------------- 第一种方法，比较方便-------------------------------
windows = reshape(x, window_size, H/window_size, window_size, W/window_size, C, B);

windows = permute(windows,[3,1,5,6,4,2]); 

windows = reshape(windows,window_size*window_size,C,[]);

end