import math

def get_best_tile_size( lowerbound, upperbound):
    """
    Get the best tile size for GPU memory
    """
    divider = 32
    while divider >= 2:
        remainer = lowerbound % divider
        if remainer == 0:
            return lowerbound
        candidate = lowerbound - remainer + divider
        if candidate <= upperbound:
            return candidate
        divider //= 2
    return lowerbound

def split_tiles(h, w):
    """
    Tool function to split the image into tiles
    @param h: height of the image
    @param w: width of the image
    @return: tile_input_bboxes, tile_output_bboxes
    """
    tile_input_bboxes, tile_output_bboxes = [], []
    tile_size = 1024
    pad = 32
    num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
    num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
    # If any of the numbers are 0, we let it be 1
    # This is to deal with long and thin images
    num_height_tiles = max(num_height_tiles, 1)
    num_width_tiles = max(num_width_tiles, 1)

    # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
    real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
    real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
    # print('1:', real_tile_height, real_tile_width)  # 640, 976
    real_tile_height = get_best_tile_size(real_tile_height, tile_size)
    real_tile_width = get_best_tile_size(real_tile_width, tile_size)
    # print('2:',real_tile_height, real_tile_width)   # 640, 992

    print(f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles*num_width_tiles} tiles. ' +
            f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

    for i in range(num_height_tiles):
        for j in range(num_width_tiles):
            # bbox: [x1, x2, y1, y2]
            # the padding is is unnessary for image borders. So we directly start from (32, 32)
            input_bbox = [
                pad + j * real_tile_width,
                min(pad + (j + 1) * real_tile_width, w),
                pad + i * real_tile_height,
                min(pad + (i + 1) * real_tile_height, h),
            ]

            # if the output bbox is close to the image boundary, we extend it to the image boundary
            output_bbox = [
                input_bbox[0] if input_bbox[0] > pad else 0,
                input_bbox[1] if input_bbox[1] < w - pad else w,
                input_bbox[2] if input_bbox[2] > pad else 0,
                input_bbox[3] if input_bbox[3] < h - pad else h,
            ]

            print(input_bbox, output_bbox)

            # scale to get the final output bbox
            output_bbox = [x * 8 if False else x // 8 for x in output_bbox]
            tile_output_bboxes.append(output_bbox)

            # indistinguishable expand the input bbox by pad pixels
            tile_input_bboxes.append([
                max(0, input_bbox[0] - pad),
                min(w, input_bbox[1] + pad),
                max(0, input_bbox[2] - pad),
                min(h, input_bbox[3] + pad),
            ])

    # print(tile_input_bboxes, tile_output_bboxes)

    return tile_input_bboxes, tile_output_bboxes



tile_input_bboxes, tile_output_bboxes = split_tiles(1344, 2016)
