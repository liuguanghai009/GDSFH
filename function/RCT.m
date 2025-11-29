function relative_coords_table = RCT (window_size,pretrained_window_size)

relative_coords_h = -(window_size - 1) : window_size-1;

relative_coords_w = -(window_size - 1) : window_size-1;

[coords_w, coords_h] = meshgrid(relative_coords_h, relative_coords_w);

relative_coords_table = cat(3,coords_h,coords_w);

if pretrained_window_size > 0

    relative_coords_table(:,:,1) = relative_coords_table(:,:,1) ./ (pretrained_window_size -1);

    relative_coords_table(:,:,2) = relative_coords_table(:,:,2) ./ (pretrained_window_size -1);

else

    relative_coords_table(:,:,1) = relative_coords_table(:,:,1) ./ (window_size -1);

    relative_coords_table(:,:,2) = relative_coords_table(:,:,2) ./ (window_size -1);

end

relative_coords_table = relative_coords_table *8;%normalize to -8, 8

relative_coords_table = sign(relative_coords_table) .* log2(abs(relative_coords_table) +1) / log2(8);

end