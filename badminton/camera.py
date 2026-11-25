"""Finite pinhole cameras. World points are in metres, image points in pixels.

Camera coordinates point right, down and forwards. A radial coefficient, when
present, operates on normalized camera coordinates, before intrinsic scaling.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation


def _points(values, dimension):
    points = np.asarray(values, dtype=float)
    if points.ndim == 0 or points.shape[-1] != dimension or not np.isfinite(points).all():
        raise ValueError(f"Expected finite points with last dimension {dimension}")
    return points


def _projection_matrix(P):
    p = np.asarray(P, dtype=float)
    if p.shape != (3, 4) or not np.isfinite(p).all():
        raise ValueError("P must be a finite 3 x 4 camera matrix")
    scale = np.max(np.abs(p[:, :3]))
    if scale == 0:
        raise ValueError("Camera has a singular left block")
    p = p / scale
    m = p[:, :3]
    if not np.isfinite(p).all() or np.linalg.cond(m) >= 1e13:
        raise ValueError("Infinite or numerically singular camera")
    # Positive focal lengths and a proper rotation fix the sign ambiguity of P.
    return p * (np.linalg.slogdet(m)[0] / np.linalg.norm(m[2]))


def project_points(P, xyz):
    """Project (...,3) points; reject points on or behind the principal plane."""
    p = _projection_matrix(P)
    xyw = _points(xyz, 3) @ p[:, :3].T + p[:, 3]
    if np.any(xyw[..., 2] <= 1e-12):
        raise ValueError("A world point is on or behind the camera plane")
    return xyw[..., :2] / xyw[..., 2, None]


def pixel_rays(P, uv):
    """Return the world camera centre (3,) and unit forward directions (...,3)."""
    p = _projection_matrix(P)
    uv = _points(uv, 2)
    pixels = np.column_stack((uv.reshape(-1, 2), np.ones(uv.size // 2)))
    rays = np.linalg.solve(p[:, :3], pixels.T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return -np.linalg.solve(p[:, :3], p[:, 3]), rays.reshape((*uv.shape[:-1], 3))


def _size(image_size):
    size = np.asarray(image_size)
    if (size.shape != (2,) or not np.isfinite(size).all()
            or np.any(size < 2) or np.any(size != np.floor(size))):
        raise ValueError("image_size must give integer width and height >= 2")
    return tuple(map(int, size))


@dataclass(frozen=True)
class Camera:
    """Centred square-pixel camera. P maps world points to ideal undistorted pixels."""

    image_size: tuple
    focal_px: float
    R: np.ndarray
    t: np.ndarray
    k1: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "image_size", _size(self.image_size))
        r = np.array(self.R, dtype=float, copy=True)
        t = np.array(self.t, dtype=float, copy=True)
        if (r.shape != (3, 3) or not np.isfinite(r).all() or np.linalg.det(r) <= 0
                or not np.allclose(r @ r.T, np.eye(3), rtol=0, atol=1e-8)):
            raise ValueError("R must be a proper orthonormal world-to-camera rotation")
        if t.shape != (3,) or not np.isfinite(t).all():
            raise ValueError("t must be a finite three-vector")
        if not np.isfinite(self.focal_px) or self.focal_px <= 0 or not np.isfinite(self.k1):
            raise ValueError("Expected positive focal length and finite radial coefficient")
        r.flags.writeable = t.flags.writeable = False
        object.__setattr__(self, "R", r)
        object.__setattr__(self, "t", t)

    @property
    def K(self):
        cx, cy = (np.asarray(self.image_size) - 1) / 2
        return np.array([[self.focal_px, 0, cx], [0, self.focal_px, cy], [0, 0, 1]])

    @property
    def P(self):
        return self.K @ np.column_stack((self.R, self.t))

    @property
    def center(self):
        return -self.R.T @ self.t

    def project(self, xyz):
        uv = project_points(self.P, xyz)
        if self.k1 == 0:
            return uv
        principal = self.K[:2, 2]
        q = (uv - principal) / self.focal_px
        kr2 = self.k1 * np.sum(q*q, axis=-1, keepdims=True)
        if np.any(1 + 3*kr2 <= 0):
            raise ValueError("World point projects outside the invertible radial branch")
        return self.focal_px * q * (1 + kr2) + principal

    def rays(self, uv):
        uv = _points(uv, 2)
        if self.k1 == 0:
            return pixel_rays(self.P, uv)
        principal = self.K[:2, 2]
        q = (uv - principal) / self.focal_px
        radii = np.linalg.norm(q, axis=-1)
        scale = np.ones(radii.shape)
        # Invert rd = ru (1+k1 ru²) on the branch through the optical axis.
        turn = np.sqrt(-1 / (3*self.k1)) if self.k1 < 0 else None
        if turn is not None and np.any(radii >= 2*turn / 3):
            raise ValueError("Pixel lies outside the invertible radial image")
        for i, radius in enumerate(radii.flat):
            if radius > 0:
                upper = turn if turn is not None else radius
                ru = brentq(lambda u: u*(1 + self.k1*u*u) - radius, 0, upper,
                            xtol=np.finfo(float).tiny)
                scale.flat[i] = ru / radius
        return pixel_rays(self.P, principal + self.focal_px * q * scale[..., None])

    def to_dict(self):
        return {"image_size": list(self.image_size), "focal_px": float(self.focal_px),
                "k1": float(self.k1), "R": self.R.tolist(), "t": self.t.tolist(),
                "K": self.K.tolist(), "P": self.P.tolist(), "center_world_m": self.center.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["image_size"], data["focal_px"], data["R"], data["t"], data["k1"])


def look_at_camera(center, target, focal_px, image_size, *, up=(0, 0, 1)):
    """Construct a camera looking at target with the given upward world reference."""
    center, target, up = (_points(v, 3) for v in (center, target, up))
    if any(v.shape != (3,) for v in (center, target, up)):
        raise ValueError("center, target and up must each be a three-vector")
    forward = target - center
    length = np.linalg.norm(forward)
    if length == 0:
        raise ValueError("Camera centre and target must differ")
    forward /= length
    right = np.cross(forward, up)
    length = np.linalg.norm(right)
    if length <= 1e-12 * np.linalg.norm(up):
        raise ValueError("up must not be zero or parallel to the viewing direction")
    right /= length
    r = np.stack((right, np.cross(forward, right), forward))
    return Camera(image_size, focal_px, r, -r @ center)


def _normalized(points):
    centered = points - points.mean(axis=0)
    spread = np.mean(np.linalg.norm(centered, axis=1))
    if spread == 0:
        raise ValueError("Coincident landmarks cannot constrain a camera")
    return centered * (np.sqrt(points.shape[1]) / spread)


def dlt_diagnostics(xyz, uv):
    """Inspect the normalized 2N x 12 DLT system; ignore the projective gauge in s1/s11."""
    xyz, uv = _points(xyz, 3), _points(uv, 2)
    if xyz.ndim != 2 or uv.ndim != 2 or len(xyz) != len(uv) or len(xyz) < 6:
        raise ValueError("At least six paired 3-D / 2-D landmarks are required")
    world = np.column_stack((_normalized(xyz), np.ones(len(xyz))))
    image = _normalized(uv)
    a = np.zeros((2 * len(xyz), 12))
    a[0::2, 0:4] = a[1::2, 4:8] = world
    a[0::2, 8:] = -image[:, :1] * world
    a[1::2, 8:] = -image[:, 1:] * world
    s = np.linalg.svd(a, compute_uv=False)
    tol = s[0] * max(a.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(s > tol))
    return {"n_points": len(xyz), "world_affine_rank": int(np.linalg.matrix_rank(world[:, :3])),
            "image_affine_rank": int(np.linalg.matrix_rank(image)),
            "design_rank": rank, "singular_values": s.tolist(),
            "identifiable_condition": float(s[0]/s[-2]) if rank >= 11 else None,
            "nullspace_gap": float(s[-2]/s[-1]) if rank == 12 else None}


def fit_camera(xyz, uv, image_size, *, fit_k1=False):
    """Fit centred square-pixel intrinsics and pose from a noncoplanar training set.

    Minimize unweighted pixel residuals. When requested, jointly fit a one-term
    radial model; the inverse stays single-valued throughout the sensor.
    """
    xyz, uv = _points(xyz, 3), _points(uv, 2)
    diag = dlt_diagnostics(xyz, uv)
    if diag["image_affine_rank"] < 2:
        raise ValueError("Collinear image landmarks cannot constrain a camera")
    if diag["world_affine_rank"] < 3 or diag["design_rank"] < 11:
        raise ValueError("Degenerate calibration geometry: include off-plane training landmarks")
    size = _size(image_size)
    principal = (np.asarray(size) - 1) / 2
    reference_focal = float(max(size))
    corner_sq = principal @ principal
    k_init = np.array([[reference_focal, 0, principal[0]], [0, reference_focal, principal[1]], [0, 0, 1]])
    ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(xyz), np.ascontiguousarray(uv),
                              k_init, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise ValueError("Could not initialize camera pose")
    initial = np.r_[rv.ravel(), tv.ravel(), 0.]
    lower, upper = np.full(7, -np.inf), np.full(7, np.inf)
    # Broad focal bounds avoid singular limits at zero and infinity.
    lower[6], upper[6] = -7., 7.
    if fit_k1:
        initial = np.r_[initial, 0.]
        # Optimize k1 * (rd at sensor corner)^2. For negative k1 the maximum
        # distorted radius is 2/(3 sqrt(-3 k1)), giving this lower bound.
        lower, upper = np.r_[lower, -4/27 + 1e-7], np.r_[upper, np.inf]

    def unpack(params):
        r = Rotation.from_rotvec(params[:3]).as_matrix()
        f = reference_focal * np.exp(params[6])
        k1 = params[7] * f*f / corner_sq if fit_k1 else 0.0
        return r, params[3:6], f, k1

    def residual(params):
        r, t, f, k1 = unpack(params)
        pts = xyz @ r.T + t
        q = pts[:, :2] / np.maximum(pts[:, 2:], 1e-8)
        predicted = f * q * (1 + k1*np.sum(q*q, axis=1, keepdims=True)) + principal
        return (predicted - uv).ravel()

    solution = least_squares(residual, initial, bounds=(lower, upper), x_scale="jac",
                             ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=1000)
    if not solution.success:
        raise ValueError(f"Camera fit did not converge: {solution.message}")
    r, t, f, k1 = unpack(solution.x)
    camera = Camera(size, f, r, t, k1)
    camera.project(xyz)  # Reject behind-camera landmarks or a folded radial branch.
    return camera
