import numpy as np

def poisson_disk_sampling(n, r, k=30, seed=42):
    # Poisson Disk Sampling in 2D in [-1, 1]^2
    # n: number of points to sample
    # r: minimum distance between points
    # k: number of attempts before rejection
    np.random.seed(seed)

    cell_size = r / np.sqrt(2)
    grid_w = int(np.ceil(2.0 / cell_size))
    grid_h = grid_w
    # grid holds indices of samples or -1
    grid = -np.ones((grid_w, grid_h), dtype=int)

    samples = []

    def point_to_grid(p):
        gx = int((p[0] + 1.0) / cell_size)
        gy = int((p[1] + 1.0) / cell_size)
        gx = min(max(gx, 0), grid_w - 1)
        gy = min(max(gy, 0), grid_h - 1)
        return gx, gy

    r2 = r * r
    # Use a generous multiplier for k to ensure we get n points if they fit
    # but still allow it to stop if the space is truly full.
    max_total_attempts = n * k * 10
    attempts = 0

    while len(samples) < n and attempts < max_total_attempts:
        attempts += 1
        candidate = np.random.uniform(-1.0, 1.0, size=2)

        cgx, cgy = point_to_grid(candidate)

        # Check neighbors within 2 cells
        ok = True
        xmin = max(cgx - 2, 0)
        xmax = min(cgx + 2, grid_w - 1)
        ymin = max(cgy - 2, 0)
        ymax = min(cgy + 2, grid_h - 1)

        for ix in range(xmin, xmax + 1):
            for iy in range(ymin, ymax + 1):
                sidx = grid[ix, iy]
                if sidx != -1:
                    dp = samples[sidx] - candidate
                    if dp[0]*dp[0] + dp[1]*dp[1] < r2:
                        ok = False
                        break
            if not ok:
                break

        if ok:
            samples.append(candidate)
            grid[cgx, cgy] = len(samples) - 1

    return np.array(samples)


def fibonacci_lattice(n, geom):
    # Distributes n points on a unit geom
    # geom: string of either "disk" or "square"
    # returns points in [-1, 1]^2
    
    phi = (1 + 5**0.5) / 2
    indices = np.arange(n)
    off = np.random.random(2)  # random offset to avoid always starting at the same point
    
    if geom == "disk":
        # Sunflower spiral
        r = np.sqrt((indices + 0.5) / n)
        theta = 2 * np.pi * indices / phi**2 + (off[0] * 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif geom == "square":
        # Fibonacci quasi-random sequence
        x = 2 * ((indices / n + off[0]) % 1) - 1
        y = 2 * ((indices * phi + off[1]) % 1) - 1
    else:
        raise ValueError(f"Unknown geom: {geom}")
        
    return np.column_stack((x, y))

def grid(n):
    # Distributes n points on a grid
    # as rectangular as possible, points in centers of cells
    # returns points in [-1, 1]^2
    
    nx = int(np.ceil(np.sqrt(n)))
    ny = int(np.ceil(n / nx))
    
    dx = 2.0 / nx
    dy = 2.0 / ny
    
    x = np.linspace(-1 + dx/2, 1 - dx/2, nx)
    y = np.linspace(-1 + dy/2, 1 - dy/2, ny)
    
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack((xv.ravel(), yv.ravel()))
    
    return points[:n]