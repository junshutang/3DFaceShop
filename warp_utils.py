import einops
import torch

def interpolate_bilinear(grid:torch.Tensor,
                         query_points:torch.Tensor,
                         indexing:str="ij") -> torch.Tensor:
    """Similar to Matlab's interp2 function.
    Finds values for query points on a grid using bilinear interpolation.
    See [1] for the original reference (Note that the tensor shape is different, etc.).
    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/interpolate_bilinear
    Parameters
    ----------
    grid : torch.Tensor [shape=(batch_size, channels, height, width)]
    query_points : torch.Tensor [shape=(batch_size, n_queries, 2)]
    indexing : str
        whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).
    Returns
    -------
    query_values : torch.Tensor [shape=(batch_size, channels, n_queries)]
    """
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")
    if grid.ndim != 4:
        raise ValueError("grid must be 4D Tensor")
    if query_points.ndim != 3:
        raise ValueError("query_points must be 3 dimensional.")

    n_queries = query_points.size(1)

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = query_points.unbind(2)

    for i, dim in enumerate(index_order):  # height -> width
        queries = unstacked_query_points[dim]  # shape=(batch_size, n_queries)
        size_in_indexing_dimension = grid.size(i+2)  # height or width

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = torch.tensor(size_in_indexing_dimension - 2, dtype=query_points.dtype).to(queries.device)
        min_floor = torch.tensor(0.0, dtype=query_points.dtype).to(queries.device)
        floor = torch.min(torch.max(min_floor, torch.floor(queries)), max_floor).long()
        floors.append(floor.view(-1))  # shape=(batch_size * n_queries)
        ceil = floor + 1
        ceils.append(ceil.view(-1))  # shape=(batch_size * n_queries)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = (queries - floor).type(grid.dtype)
        min_alpha = torch.tensor(0.0, dtype=grid.dtype).to(queries.device)
        max_alpha = torch.tensor(1.0, dtype=grid.dtype).to(queries.device)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)  # shape=(batch_size, n_queries)

        # Expand alpha to [b, 1, n] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = torch.unsqueeze(alpha, 1)  # shape=(batch_size, 1, n_queries)
        alphas.append(alpha)

    batch_size, channels, height, width = grid.size()
    flattened_grid = einops.rearrange(grid, 'b c h w -> (b h w) c')
    batch_indice = torch.arange(batch_size).repeat(n_queries, 1).t().reshape(-1).to(grid.device)  # [0, ..., 0, 1, ..., 1, 2, ...]

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords:torch.Tensor, x_coords:torch.Tensor):
        gathered_values = grid[batch_indice, :, y_coords, x_coords]  # shape=(batch_size * n_queries, channels)
        return einops.rearrange(gathered_values, '(b q) c -> b c q', b=batch_size, q=n_queries)  # shape=(batch_size, channels, n_queries)

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1])
    top_right = gather(floors[0], ceils[1])
    bottom_left = gather(ceils[0], floors[1])
    bottom_right = gather(ceils[0], ceils[1])

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp

def dense_image_warp(image:torch.Tensor, flow:torch.Tensor) -> torch.Tensor:
    """Image warping using per-pixel flow vectors.
    See [1] for the original reference (Note that the tensor shape is different, etc.).
    [1] https://www.tensorflow.org/addons/api_docs/python/tfa/image/dense_image_warp
    Parameters
    ----------
    image : torch.Tensor [shape=(batch, channels, height, width)]
    flow : torch.Tensor [shape=(batch, 2, height, width)]
    Returns
    -------
    warped_image : torch.Tensor [shape=(batch, channels, height, width)]
    """
    batch_size, channels, height, width = image.shape
    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    y_range = torch.arange(0., height, device=image.device, requires_grad=False)
    x_range = torch.arange(0., width, device=image.device, requires_grad=False)
    y_grid, x_grid = torch.meshgrid(y_range, x_range)
    stacked_grid = torch.stack((y_grid, x_grid), dim=0)  # shape=(2, height, width)
    batched_grid = stacked_grid.unsqueeze(0)  # shape=(1, 2, height, width)
    query_points_on_grid = batched_grid - flow  # shape=(batch_size, 2, height, width)
    query_points_flattened = einops.rearrange(query_points_on_grid, 'b x h w -> b (h w) x')  # shape=(batch_size, height * width, 2)
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = interpolate_bilinear(image, query_points_flattened)  # shape=(batch_size, channels, n_queries)
    interpolated = einops.rearrange(interpolated, 'b c (h w) -> b c h w', h=height, w=width)
    return interpolated